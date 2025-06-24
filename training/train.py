'''
### Baseline Architecture Notes (DEL later): 3D Conv Autoencoder

```
Input: (B, C, T, H, W)
🔽 Conv3D layers
🔽 ReLU + MaxPool3D
→ Latent Space
🔼 ConvTranspose3D layers
🔼 ReLU + Sigmoid
Output: Reconstructed clip (B, C, T, H, W)
```

- **Loss Function**: MSE (or SSIM later)

'''