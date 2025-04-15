class Flex2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1, padding=0, config=None):
        super().__init__()
        """
        # in dimensions: in this case [C, H, W]
        # --------
        # logits_mechanism: THRESHOLD or SPATIAL_ATTENTION_(1-3)
        # masking_mechanism: "SIGMOID", "STE", "SIGMOID_SMOOTHED", "STE_SIGMOID"
        # num_spatial_attention_block: int
        # logits_use_batchnorm: bool

        # about parameter vs variable:
        variable is almost deprecated and works the same as just plain tensor. And a Parameters is a specific Tensor that is marked as being a parameter from an nn.Module and so will be returned when calling .parameters() on this Module.
        """
        # -------- set configs --------
        # assert config, "Missing config file for Flex2D"
        # self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        # -------- Initialize layers --------
        self.flex_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.flex_pool = MaxPool2d(kernel_size, stride, padding)
        self.bn_logits = nn.BatchNorm2d(self.out_channels)

        # -------- Initialize monitored variables --------
        self.homogeneity = 0  # for monitoring the binariness of the mask later on
        self.conv_ratio = 0  # for later updating
        self.cp_identity_matrix = None  # store the matrix indicating the channel pool identity

    def channel_interpolate(self, tensor, out_channels):
        """
        Interpolates a tensor along the channel axis. This is for addressing the mismatch in the number of channels between the output of the maxpool layer and the output of the convolutional layer.
        """
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = F.interpolate(tensor, [out_channels, tensor.size(3)], mode="bilinear")
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor
    
    def channel_wise_maxpool(self, tensor_1, tensor_2):
        """
        Take two tensors of identical shape and return a tensor of the same shape using element-wise max pooling.
        Also returns the ratio of values from tensor_2 to the total.
        """
        assert tensor_1.shape == tensor_2.shape, "tensor_1 and tensor_2 must have the same shape"

        joint = cat([tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)], dim=-1)
        pooled, indices = torch.max(joint, dim=-1)

        # count values are from conv (tensor 2)
        with torch.no_grad():
            conv_ratio = (indices == 1).sum().item() / tensor_1.numel()
            cp_identity_matrix = (indices == 1).int()

        return pooled, conv_ratio, cp_identity_matrix

    def forward(self, x):
        """
        threshold can only be initialized when the output dimensions are known
        """
        # -------- make the raw conv and pool --------
        t_flex_pool = self.flex_pool(x)
        t_flex_conv = self.flex_conv(x)
        t_flex_pool = self.channel_interpolate(t_flex_pool, self.out_channels)

        # -------- get the binary mask --------
        output, self.conv_ratio, self.cp_identity_matrix = self.channel_wise_maxpool(t_flex_pool, t_flex_conv)
        return output
