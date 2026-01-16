# 文件路径: legged_gym/envs/base/terrain.py
# 这是增强版 Terrain，支持 heightfield 和 trimesh 输出（兼容你的原始实现）
import numpy as np
from isaacgym import terrain_utils

class Terrain:
    def __init__(self, cfg, num_robots):
        """
        cfg: LeggedRobotCfg.terrain 实例
        num_robots: 环境总数 (用于 num_rows * num_cols)
        """
        # ------- 安全地形参数（强制覆盖）-------
        cfg.horizontal_scale = 0.1      # 每格10cm（从0.05修改）
        cfg.vertical_scale = 0.01       # 高度1cm
        cfg.terrain_length = 8.0        # 每个环境 3m
        cfg.terrain_width = 8.0
        cfg.border_size = 1.0           # 只有10格的边框
        cfg.num_rows = 6                # 环境矩阵缩小
        cfg.num_cols = 6

        # -----------------------------------
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            # nothing to build
            self.height_field_raw = None
            self.vertices = None
            self.triangles = None
            return

        # env sizes
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        # cumulative proportions
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # number of sub-terrains
        self.cfg.num_sub_terrains = int(cfg.num_rows * cfg.num_cols)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # pixels per env
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # height field raw (integer)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # build sub-terrains and map
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        # heightsamples for later use
        self.heightsamples = self.height_field_raw

        # if trimesh requested, convert
        if self.type == "trimesh":
            verts, tris = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                cfg.horizontal_scale,
                cfg.vertical_scale,
                cfg.slope_treshold
            )
            # ensure arrays (numpy)
            self.vertices = np.asarray(verts)
            self.triangles = np.asarray(tris)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            choice = np.random.uniform(0., 1.)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, self.cfg.num_rows)
                choice = (j / max(1, self.cfg.num_cols)) + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # user selected terrain type: expects cfg.terrain_kwargs to be a dict with key 'type' etc.
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            sub = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale
            )
            # eval the name in current module scope of terrain_utils
            eval(f"terrain_utils.{terrain_type}")(sub, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(sub, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )

        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty

        # choose type according to proportions
        if choice < self.proportions[0]:
            if choice < (self.proportions[0] / 2.):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        # ---- 写 height field 到大图 ----
        start_x = self.border + row * self.length_per_env_pixels
        end_x   = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y   = self.border + (col + 1) * self.width_per_env_pixels

        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # ---- 机器人放在整个地形的中心 ----
        total_x = self.cfg.num_rows * self.env_length
        total_y = self.cfg.num_cols * self.env_width

        center_x = total_x / 2
        center_y = total_y / 2

        # 计算该 sub-terrain 的中间位置 (用于计算中心高度)
        t_center = terrain.height_field_raw[
            int(terrain.length * 0.4):int(terrain.length * 0.6),
            int(terrain.width * 0.4):int(terrain.width * 0.6)
        ]
        center_z = np.max(t_center) * terrain.vertical_scale

        # 所有 env 都放在场地中央
        self.env_origins[row, col] = [center_x, center_y, center_z]

def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2:center_x + x2, center_y - y2:center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1:center_x + x1, center_y - y1:center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
