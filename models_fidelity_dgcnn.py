import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=10):
        super(DynamicGraphConv, self).__init__()
        self.k = k
        # Properly initialize conv layer with correct input channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
    def construct_graph(self, x):
        """
        Construct dynamic graph using k-nearest neighbors
        x shape: [batch_size, num_points, features]
        """
        batch_size, num_points, num_features = x.size()
        
        # Calculate pairwise distances
        inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [batch_size, num_points, num_points]
        xx = torch.sum(x**2, dim=2, keepdim=True)  # [batch_size, num_points, 1]
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [batch_size, num_points, num_points]
        
        # Find k nearest neighbors
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # [batch_size, num_points, k]
        return idx
        
    def forward(self, x):
        """
        x shape: [batch_size, features, num_points]
        """
        batch_size, num_features, num_points = x.size()
        
        # Transpose to [batch_size, num_points, features]
        x = x.transpose(1, 2).contiguous()
        
        # Construct dynamic graph
        idx = self.construct_graph(x)  # [batch_size, num_points, k]
        
        # Get features from k-nearest neighbors
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighbor_features = x.view(batch_size*num_points, -1)[idx, :]
        neighbor_features = neighbor_features.view(batch_size, num_points, self.k, num_features)
        x = x.view(batch_size, num_points, 1, num_features).repeat(1, 1, self.k, 1)
        
        # Edge features
        edge_features = torch.cat((x, neighbor_features-x), dim=3)  # [batch_size, num_points, k, 2*features]
        edge_features = edge_features.permute(0, 3, 1, 2)  # [batch_size, 2*features, num_points, k]
        
        # Apply convolution
        out = self.conv(edge_features)  # [batch_size, out_channels, num_points, k]
        out = out.max(dim=-1, keepdim=False)[0]  # [batch_size, out_channels, num_points]
        
        return out

class FidelityAwareMultimodalDGCNN(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        self.text_dim = hyp_params.orig_d_l
        self.audio_dim = hyp_params.orig_d_a
        self.visual_dim = hyp_params.orig_d_v
        # shared latent size per modality before DGCNN
        self.hidden_dim = 256
        self.embed_dim = 64  # Dimension for fidelity embeddings
        self.output_dim = hyp_params.output_dim
        
        # Project features using properly oriented Conv1d layers
        self.proj_text = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.proj_audio = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1)
        )
        self.proj_visual = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dynamic Graph CNN layers
        self.dgcnn_text = DynamicGraphConv(self.hidden_dim, self.hidden_dim)
        self.dgcnn_audio = DynamicGraphConv(self.hidden_dim, self.hidden_dim)
        self.dgcnn_visual = DynamicGraphConv(self.hidden_dim, self.hidden_dim)
        
        # Fidelity embeddings
        self.U_text = nn.Parameter(torch.randn(self.embed_dim))
        self.V_text = nn.Parameter(torch.randn(self.embed_dim))
        self.U_audio = nn.Parameter(torch.randn(self.embed_dim))
        self.V_audio = nn.Parameter(torch.randn(self.embed_dim))
        self.U_visual = nn.Parameter(torch.randn(self.embed_dim))
        self.V_visual = nn.Parameter(torch.randn(self.embed_dim))
        
        # Feature fusion layers
        fusion_dim = self.hidden_dim * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 4, self.output_dim)
        )

        # verification hook
        print("projection layers output dim = 256")

    def get_beta_params(self, U, V):
        """Compute Beta distribution parameters"""
        mu = 0.5 + 0.5 * (U @ V) / (torch.norm(U) * torch.norm(V))
        nu = torch.norm(U) * torch.norm(V)
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def get_fidelity_scores(self, text, audio, visual):
        """Calculate fidelity scores using Beta distribution"""
        # Get Beta parameters for each modality
        alpha_t, beta_t = self.get_beta_params(self.U_text, self.V_text)
        alpha_a, beta_a = self.get_beta_params(self.U_audio, self.V_audio)
        alpha_v, beta_v = self.get_beta_params(self.U_visual, self.V_visual)
        
        # Calculate expected fidelity and variance
        def get_moments(alpha, beta):
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            return mean, var
        
        E_t, var_t = get_moments(alpha_t, beta_t)
        E_a, var_a = get_moments(alpha_a, beta_a)
        E_v, var_v = get_moments(alpha_v, beta_v)
        
        # Compute fidelity-aware gating weights
        scores = torch.tensor([
            E_t / torch.sqrt(var_t),
            E_a / torch.sqrt(var_a),
            E_v / torch.sqrt(var_v)
        ])
        
        return F.softmax(scores, dim=0)

    def forward(self, text, audio, visual):
        # Input shape: [batch, features, time]
        # Transpose to [batch, time, features] for linear projection
        text = text.transpose(1, 2)    # [batch, time, 768]
        audio = audio.transpose(1, 2)  # [batch, time, 40]
        visual = visual.transpose(1, 2) # [batch, time, 2048]
        
        # Project features to hidden_dim
        text = self.proj_text(text)    # [batch, time, 128]
        audio = self.proj_audio(audio) # [batch, time, 128]
        visual = self.proj_visual(visual) # [batch, time, 128]
        
        # Transpose back to [batch, features, time] for DGCNN
        text = text.transpose(1, 2)    # [batch, 128, time]
        audio = audio.transpose(1, 2)  # [batch, 128, time]
        visual = visual.transpose(1, 2) # [batch, 128, time]
        
        # Apply DGCNN
        text = self.dgcnn_text(text)    # [batch, 128, time]
        audio = self.dgcnn_audio(audio) # [batch, 128, time]
        visual = self.dgcnn_visual(visual) # [batch, 128, time]
        
        # Global pooling over time dimension
        text = text.mean(dim=2)    # [batch, 128]
        audio = audio.mean(dim=2)  # [batch, 128]
        visual = visual.mean(dim=2) # [batch, 128]
        
        # Get fidelity weights
        weights = self.get_fidelity_scores(text, audio, visual)
        
        # Weighted pooling
        text_weighted = weights[0] * text     # [batch, 128]
        audio_weighted = weights[1] * audio   # [batch, 128]
        visual_weighted = weights[2] * visual # [batch, 128]
        
        # Concatenate for fusion
        fused = torch.cat([text_weighted, audio_weighted, visual_weighted], dim=1) # [batch, 384]
        
        # Final classification
        return self.fusion_layer(fused)


if __name__ == '__main__':
    # Example hyperparameters
    class HyperParams:
        def __init__(self):
            self.orig_d_l = 300    # text feature dimension
            self.orig_d_a = 74     # audio feature dimension
            self.orig_d_v = 35     # visual feature dimension
            self.output_dim = 8    # for IEMOCAP (4 emotions * 2 classes each)

    # Create example inputs
    batch_size = 32
    seq_len = 50
    
    # Initialize model
    hyp_params = HyperParams()
    model = FidelityAwareMultimodalDGCNN(hyp_params)
    print("\nModel Architecture:")
    print(model)
    
    # Create dummy input tensors
    text = torch.randn(batch_size, hyp_params.orig_d_l, seq_len)    # [batch, features, sequence]
    audio = torch.randn(batch_size, hyp_params.orig_d_a, seq_len)
    visual = torch.randn(batch_size, hyp_params.orig_d_v, seq_len)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    text = text.to(device)
    audio = audio.to(device)
    visual = visual.to(device)

    print("\nInput shapes:")
    print(f"Text input shape: {text.shape}")
    print(f"Audio input shape: {audio.shape}")
    print(f"Visual input shape: {visual.shape}")

    # Forward pass
    with torch.no_grad():
        output, fidelity_weights = model(text, audio, visual)

    print("\nOutput shapes:")
    print(f"Model output shape: {output.shape}")
    print(f"Fidelity weights shape: {fidelity_weights.shape}")
    print("\nFidelity weights for each modality:")
    print(f"Text: {fidelity_weights[0]:.4f}")
    print(f"Audio: {fidelity_weights[1]:.4f}")
    print(f"Visual: {fidelity_weights[2]:.4f}")

    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
