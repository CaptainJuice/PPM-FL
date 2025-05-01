import torch

def add_uniform_noise_to_point_clouds(point_clouds, noise_percentage=0.1):
    n, m, _ = point_clouds.shape
    num_noise_points = int(m * noise_percentage)
    torch.manual_seed(2026)

    noisy_point_clouds = []
    for i in range(n):
        point_cloud = point_clouds[i]
        min_vals = torch.min(point_cloud, dim=0)[0]
        max_vals = torch.max(point_cloud, dim=0)[0]
        noise_points = torch.rand(num_noise_points, 3) * (max_vals - min_vals) + min_vals
        replace_indices = torch.randperm(m)[:num_noise_points]
        new_point_cloud = point_cloud.clone()
        new_point_cloud[replace_indices] = noise_points

        noisy_point_clouds.append(new_point_cloud)

    return torch.stack(noisy_point_clouds, dim=0)

# if __name__=='__main__':
#     n = 5
#     m = 100
#     point_clouds = torch.rand(n, m, 3)
#     noisy_point_clouds = add_uniform_noise_to_point_clouds(point_clouds)
#     print(f"原始点云形状: {point_clouds.shape}")
#     print(f"添加噪声后的点云形状: {noisy_point_clouds.shape}") 