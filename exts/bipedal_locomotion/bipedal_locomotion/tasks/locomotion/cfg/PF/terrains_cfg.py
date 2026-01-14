from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)

#############################
# 粗糙地形配置 / Rough Terrain Configuration
#############################

# 盲视粗糙地形配置 - 用于无视觉传感器的训练
# Blind rough terrain configuration - for training without vision sensors
BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,                        # 随机种子确保可重复性 / Random seed for reproducibility
    size=(8.0, 8.0),               # 每个地形块大小 8x8米 / Each terrain tile size 8x8 meters
    border_width=20.0,              # 边界宽度 / Border width
    num_rows=10,                    # 地形行数 / Number of terrain rows
    num_cols=16,                    # 地形列数 / Number of terrain columns
    horizontal_scale=0.1,           # 水平分辨率 / Horizontal resolution
    vertical_scale=0.005,           # 垂直分辨率 / Vertical resolution
    slope_threshold=0.75,           # 斜率阈值 / Slope threshold
    use_cache=True,                 # 使用缓存加速生成 / Use cache for faster generation
   
    # 子地形配置 - 定义不同类型的地形
    # Sub-terrain configurations - define different types of terrain
    sub_terrains={
        # 平地 (25%占比) / Flat terrain (25% proportion)
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        
        # 波浪地形 (25%占比) / Wave terrain (25% proportion)  
        "waves": HfWaveTerrainCfg(
            proportion=0.25, 
            amplitude_range=(0.01, 0.06),      # 波浪幅度范围 [m] / Wave amplitude range [m]
            num_waves=10,                      # 波浪数量 / Number of waves
            border_width=0.25                  # 边界宽度 / Border width
        ),
        
        # 随机格子地形 (25%占比) / Random grid terrain (25% proportion)
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, 
            grid_width=0.15,                   # 格子宽度 / Grid width
            grid_height_range=(0.01, 0.04),    # 格子高度范围 [m] / Grid height range [m]
            platform_width=2.0                 # 平台宽度 / Platform width
        ),
        
        # 随机粗糙地形 (25%占比) / Random rough terrain (25% proportion)
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.25, 
            noise_range=(0.01, 0.06),          # 噪声高度范围 [m] / Noise height range [m]
            noise_step=0.01,                   # 噪声步长 / Noise step
            border_width=0.25                  # 边界宽度 / Border width
        ),
    },
    
    curriculum=True,                    # 启用课程学习 / Enable curriculum learning
    difficulty_range=(0.0, 1.0),       # 难度范围 0-1 / Difficulty range 0-1
)

# 盲视粗糙地形测试配置 - 用于策略评估
# Blind rough terrain play configuration - for policy evaluation
BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,                         # 减少行数用于测试 / Reduced rows for testing
    num_cols=4,                         # 减少列数用于测试 / Reduced columns for testing
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    
    sub_terrains={
        # 只保留三种地形类型 / Only keep three terrain types
        "waves": HfWaveTerrainCfg(
            proportion=0.33,                # 33%占比 / 33% proportion
            amplitude_range=(0.01, 0.06), 
            num_waves=10, 
            border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2,                 # 20%占比 / 20% proportion
            grid_width=0.33,                # 更大的格子 / Larger grid
            grid_height_range=(0.01, 0.04), 
            platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.34,                # 34%占比 / 34% proportion
            noise_range=(0.01, 0.06), 
            noise_step=0.01, 
            border_width=0.25
        ),
    },
    
    curriculum=False,                   # 测试时不使用课程学习 / No curriculum for testing
    difficulty_range=(1.0, 1.0),       # 固定最高难度 / Fixed maximum difficulty
)


##################################
# 困难粗糙地形配置 / Hard Rough Terrain Configuration
##################################

BLIND_HARD_ROUGH_TERRAINS_CFG = BLIND_ROUGH_TERRAINS_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.02

BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG = BLIND_ROUGH_TERRAINS_PLAY_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_step = 0.02

##############################
# 楼梯地形配置 / Stairs Terrain Configuration
##############################

# 楼梯地形训练配置 - 用于训练爬楼梯能力
# Stairs terrain training configuration - for training stair climbing ability
STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),                  # 更大的地形块适合楼梯 / Larger terrain tiles for stairs
    border_width=20.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    
    sub_terrains={
        # 金字塔楼梯 (40%占比) / Pyramid stairs (40% proportion)
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),    # 台阶高度范围 5-20cm / Step height range 5-20cm
            step_width=0.3,                    # 台阶宽度 30cm / Step width 30cm
            platform_width=3.0,                # 平台宽度 3m / Platform width 3m
            border_width=1.0,                  # 边界宽度 / Border width
            holes=False,                       # 不添加洞 / No holes
        ),
        
        # 倒金字塔楼梯 (40%占比) / Inverted pyramid stairs (40% proportion)
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),    # 下降台阶 / Descending steps
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        
        # 金字塔斜坡 (10%占比) / Pyramid slope (10% proportion)
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.4),            # 斜率范围 0-40% / Slope range 0-40%
            platform_width=2.0, 
            border_width=0.25
        ),
        
        # 倒金字塔斜坡 (10%占比) / Inverted pyramid slope (10% proportion)
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.4), 
            platform_width=2.0, 
            border_width=0.25
        ),
    },
    
    curriculum=True,                        # 启用课程学习 / Enable curriculum learning
    difficulty_range=(0.0, 1.0),
)

STAIRS_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)
