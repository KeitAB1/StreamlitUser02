import numpy as np

class SteelPlateStackingObjectives:
    def __init__(self, plates, heights, delivery_times, batches, Dki, area_positions, inbound_point, outbound_point, horizontal_speed, vertical_speed):
        self.plates = plates
        self.num_plates = len(plates)
        self.heights = heights
        self.delivery_times = delivery_times
        self.batches = batches
        self.Dki = Dki
        self.area_positions = area_positions
        self.inbound_point = inbound_point
        self.outbound_point = outbound_point
        self.horizontal_speed = horizontal_speed
        self.vertical_speed = vertical_speed
        self.max_height = 3000  # 设置全局最大高度限制

    # 目标函数1：最小化翻垛次数和翻转惩罚
    def minimize_stack_movements_and_turnover(self, particle_positions, weight_movement=2.0, weight_turnover=3.0):
        num_movements = 0
        total_turnover = 0
        batch_turnover = 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = self.plates[plate_idx, 2]
            current_height = self.heights[area]

            # 限制库区的最大堆叠高度，避免翻转次数过多
            if current_height + plate_height > self.max_height:
                num_movements += 1

            self.heights[area] += plate_height

        for i in range(len(particle_positions)):
            for j in range(i + 1, len(particle_positions)):
                time_diff = abs(self.delivery_times[i] - self.delivery_times[j])
                total_turnover += time_diff

                if self.batches[i] != self.batches[j]:
                    batch_turnover += 1

        # 增加翻转次数的惩罚
        combined_score = weight_movement * num_movements + weight_turnover * (total_turnover + batch_turnover)
        return combined_score + (num_movements ** 2)

    # 目标函数2：最小化出库能耗与时间
    def minimize_outbound_energy_time_with_batch(self, particle_positions):
        total_time_energy = 0

        sorted_batches = sorted(set(self.batches), key=lambda x: int(x[1:]))
        plate_indices_by_batch = {batch: [] for batch in sorted_batches}

        for plate_idx, plate in enumerate(self.plates):
            batch = plate[4]
            plate_indices_by_batch[batch].append(plate_idx)

        def calculate_distance(x1, y1, x2, y2):
            return abs(x2 - x1), abs(y2 - y1)

        def calculate_pick_time(plate_height):
            return plate_height / self.vertical_speed

        def calculate_flip_time(plate_idx):
            area = particle_positions[plate_idx]
            current_height = self.heights[area]
            plate_height = self.plates[plate_idx, 2]

            if current_height > plate_height:
                n_flip = min(int(current_height // plate_height), 3)  # 限制最大翻转次数
                return n_flip * 10
            else:
                return 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = self.plates[plate_idx, 2]
            x, y = self.area_positions[area][plate_idx % len(self.area_positions[area])]

            inbound_horizontal_dist, inbound_vertical_dist = calculate_distance(x, y, self.inbound_point[0], self.inbound_point[1])
            inbound_time = (inbound_horizontal_dist / self.horizontal_speed) + (inbound_vertical_dist / self.vertical_speed)

            self.heights[area] += plate_height
            total_time_energy += inbound_time

        for batch in sorted_batches:
            for plate_idx in plate_indices_by_batch[batch]:
                area = particle_positions[plate_idx]
                plate_height = self.plates[plate_idx, 2]
                x, y = self.area_positions[area][plate_idx % len(self.area_positions[area])]

                outbound_horizontal_dist, outbound_vertical_dist = calculate_distance(x, y, self.outbound_point[0], self.outbound_point[1])
                outbound_time = (outbound_horizontal_dist / self.horizontal_speed) + (outbound_vertical_dist / self.vertical_speed)
                pick_time = calculate_pick_time(plate_height)
                flip_time = calculate_flip_time(plate_idx)

                self.heights[area] -= plate_height
                total_time_energy += (outbound_time + pick_time + flip_time)

        # 增加高度平衡的惩罚，确保高度不超出设定值
        height_penalty = np.sum([max(0, height - self.max_height) for height in self.heights])
        return total_time_energy + height_penalty * 1000  # 增加不均衡的惩罚

    # 目标函数3：最大化库存均衡度，加入更严格的均衡性惩罚
    def maximize_inventory_balance_v2(self, particle_positions):
        total_variance = 0
        total_volume = np.sum(self.plates[:, 0] * self.plates[:, 1] * self.plates[:, 2])
        num_positions = len(self.Dki)
        mean_volume_per_position = total_volume / num_positions
        area_volumes = np.zeros(num_positions)

        for plate_idx, position in enumerate(particle_positions):
            plate_volume = self.plates[plate_idx][0] * self.plates[plate_idx][1] * self.plates[plate_idx][2]
            area_volumes[position] += plate_volume

        for j in range(num_positions):
            total_variance += (area_volumes[j] - mean_volume_per_position) ** 2

        # 增加高度不均衡的惩罚
        for area in range(num_positions):
            if self.heights[area] > self.max_height:
                total_variance += 10000  # 增加惩罚

        return total_variance / num_positions

    # 目标函数4：空间利用率最大化，加入惩罚机制
    def maximize_space_utilization_v3(self, particle_positions, alpha_1=1.0, epsilon=1e-6):
        total_space_utilization = 0
        for i in range(len(self.Dki)):
            used_volume = 0
            max_volume = self.Dki[i]

            for j in range(len(self.plates)):
                if particle_positions[j] == i:
                    plate_volume = self.plates[j][0] * self.plates[j][1] * self.plates[j][2]
                    used_volume += plate_volume

            if used_volume > 0:
                utilization = alpha_1 * max((max_volume - used_volume), epsilon) / used_volume
                total_space_utilization += utilization

            # 如果某个库区利用率过低，增加惩罚
            if used_volume < 0.5 * max_volume:  # 设置利用率阈值
                total_space_utilization += 10000  # 增加惩罚

        return total_space_utilization


class ContainerLoadingObjectives:
    def __init__(self, items, container_dimensions):
        self.items = items
        self.num_items = len(items)
        self.container_dimensions = container_dimensions
        self.container_volume = np.prod(container_dimensions)
        self.max_weight = 14000  # 假设集装箱最大承载重量为24吨（可根据实际调整）

    # 目标函数1：最大化空间利用率
    def maximize_space_utilization(self, positions):
        total_volume = 0

        for i, position in enumerate(positions):
            length, width, height = self.items[i][:3]
            item_volume = length * width * height
            total_volume += item_volume

        space_utilization = total_volume / self.container_volume
        return -space_utilization  # 负值，因为需要最大化

    # 目标函数2：最小化重心偏差
    def minimize_center_of_gravity_deviation(self, positions):
        total_weight = 0
        weighted_x_sum = 0
        weighted_y_sum = 0
        weighted_z_sum = 0

        container_length, container_width, container_height = self.container_dimensions

        for i, position in enumerate(positions):
            length, width, height, weight = self.items[i][:4]
            x, y, z = position  # 假设 position 给出了每个物体在集装箱内的放置位置
            total_weight += weight
            weighted_x_sum += weight * (x + length / 2)
            weighted_y_sum += weight * (y + width / 2)
            weighted_z_sum += weight * (z + height / 2)

        if total_weight == 0:
            return np.inf  # 防止除零错误

        center_of_gravity = (
            weighted_x_sum / total_weight,
            weighted_y_sum / total_weight,
            weighted_z_sum / total_weight,
        )
        ideal_center_of_gravity = (
            container_length / 2,
            container_width / 2,
            container_height / 2,
        )
        deviation = np.linalg.norm(np.array(center_of_gravity) - np.array(ideal_center_of_gravity))
        return deviation

    # 目标函数3：约束超载
    def enforce_weight_limit(self, positions):
        total_weight = sum(self.items[i][3] for i in range(len(positions)))
        return max(0, total_weight - self.max_weight) * 1000  # 超重部分的惩罚

    # 计算总适应度
    def evaluate(self, positions):
        space_utilization_score = self.maximize_space_utilization(positions)
        gravity_deviation_score = self.minimize_center_of_gravity_deviation(positions)
        weight_penalty = self.enforce_weight_limit(positions)

        return (
            space_utilization_score * 0.5
            + gravity_deviation_score * 0.3
            + weight_penalty * 0.2
        )
