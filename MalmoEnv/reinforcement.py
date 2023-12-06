import numpy as np

import utils

ATTACK_RANGE = 3
GRID_SIZE = 5
NEGATIVE_WALL_COORDINATE = -2.075
POSITIVE_WALL_COORDINATE = 3.075
AGENT_HEIGHT = 1.6
CHICKEN_HEIGHT = 0.5


class State:
    def __init__(self, data):
        self.data = data
        self.xz_delta = self._calculate_xz_delta()
        self.pitch = self._calculate_pitch()
        self.yaw = self._calculate_yaw()

    def _calculate_xz_delta(self):
        entities_dict = self.data["entities"]
        xz_delta = (utils.decimal_round(entities_dict[1]['x'], 0.1) - utils.decimal_round(self.data['XPos'], 0.1),
                     utils.decimal_round(entities_dict[1]['z'], 0.1) - utils.decimal_round(self.data['ZPos'], 0.1))

        return xz_delta

    def _calculate_pitch(self):
        pitch = utils.integer_round(self.data['Pitch'], 1)
        return pitch

    def _calculate_yaw(self):
        yaw = utils.integer_round((360 + self.data['Yaw']) % 360, 1)
        return yaw

    def get_xz_delta(self):
        return self.xz_delta

    def get_pitch(self):
        return self.pitch

    def get_yaw(self):
        return self.yaw

    def get_coordinates(self):
        return round(self.data['XPos'], 3), round(self.data['ZPos'], 3)

    def get_distance_from_chicken(self):
        return utils.decimal_round(utils.distance(self.xz_delta), 0.25, 2)
    
    """
    Negative angle means the agent needs to pan to the left to see the target.
    Positive angle means the agent needs to pan to the right to see the target.
    """
    def get_signed_looking_angle_from_chicken(self):
        yaw = self.get_yaw()
        target_angle = utils.calculate_target_angle(self.get_xz_delta())
        yaw_difference = utils.integer_round(-1 * ((yaw - target_angle + 180) % 360 - 180), 10)

        return yaw_difference

    def get_pitch_difference_from_chicken(self):
        pitch = self.get_pitch() if self.get_pitch() >= 0 else -5
        angle_of_depression = utils.angle_of_depression(AGENT_HEIGHT, utils.distance(self.get_xz_delta()))
        return -1 * utils.integer_round(pitch - angle_of_depression, 5)


    def get_is_near_wall(self):
        xPos = self.data['XPos']
        zPos = self.data['ZPos']
        positiveWallCoordinate = -2.075
        negativeWallCoordinate = 3.075
        if abs(xPos - positiveWallCoordinate) <= 0.5 or abs(xPos - negativeWallCoordinate) <= 0.5 or \
                abs(zPos - positiveWallCoordinate) <= 0.5 or abs(zPos - negativeWallCoordinate) <= 0.5:
            return -1
        else:
            return 1


    def to_state(self):
        return np.array((self.get_distance_from_chicken(),
                         self.get_signed_looking_angle_from_chicken(),
                         self.get_pitch_difference_from_chicken(),
                         self.get_is_near_wall()))


class Reward:
    def __init__(self, data, state: State, action: int, previous_coordinates, current_damage_dealt):
        self.data = data
        self.state = state
        self.action = action
        self.previous_coordinates = previous_coordinates
        self.current_damage_dealt = current_damage_dealt

        self.reward = -10

    def _penalize_for_looking_above_eye_level(self):
        # penalize for looking above eye-level (useless action)
        # and snap it back to eye-level.
        if self.state.get_pitch() < 0:
            self.reward -= 10
            print("penalized -10 for: looking above eye-level")
            return "setPitch 0"

    def _penalize_for_pitching_down_when_already_looking_straight_down(self):
        # penalize if it tried to use "pitch 1" when it's already looking straight down (useless action)
        if self.state.get_pitch() == 90 and self.action == 1:
            self.reward -= 10
            print("penalized -10 for: tried pitching down when already looking straight down")

    def _penalize_for_attacking(self):
        # penalize if it used "attack 1" - don't want agent to spam it in order to stand still and farm rewards
        if (self.action == 5):
            self.reward -= 2
            print("    Attack penalty: -2")

    def _penalize_for_getting_stuck(self):
        # penalize if it used "move 1" but didn't go anywhere - getting stuck = bad
        if self.action == 0 and self.previous_coordinates == (round(self.data['XPos'], 3), round(self.data['ZPos'], 3)):
            self.reward -= 150
            print("penalized -150 for: getting stuck")

    def _penalize_near_wall(self):
        if self.state.get_is_near_wall() == -1:
            print("    Wall penalty: -7")
            self.reward += -7
        else:
            print()

    def _calculate_distance_from_chicken_reward(self):
        # closer = better reward, too far = penalized
        distance_reward = (-0.5 * self.state.get_distance_from_chicken()) if self.state.get_distance_from_chicken() > ATTACK_RANGE \
            else 2 * (ATTACK_RANGE - self.state.get_distance_from_chicken())
        self.reward += distance_reward
        print(f"    Dis reward: {distance_reward}")

    def _calculate_yaw_from_chicken_reward(self):
        # # reward for looking in the direction of chicken, penalize for not
        yaw_difference = abs(self.state.get_signed_looking_angle_from_chicken())
        if yaw_difference >= 90:
            yaw_reward = -1 * yaw_difference / 72
        else:
            yaw_reward = ((180 - yaw_difference) / 72)
            if yaw_difference <= 30:
                self._calculate_pitch_from_chicken_reward()
        print(f"    Yaw reward: {yaw_reward}")
        self.reward += yaw_reward

    def _calculate_pitch_from_chicken_reward(self):
        distance_weight = 1.2
        angular_weight = 1.6

        # if greater than these, starts getting penalized
        distance_threshold = 3.0
        angular_threshold = 15.0

        distance_reward = max(0, distance_threshold - self.state.get_distance_from_chicken()) / distance_threshold
        angular_reward = max(0, angular_threshold - abs(self.state.get_pitch_difference_from_chicken())) / angular_threshold

        pitch_reward = distance_weight * distance_reward + angular_weight * angular_reward
        misalignment_penalty = max(0, abs(self.state.get_pitch_difference_from_chicken()) - angular_threshold) / angular_threshold
        pitch_reward -= misalignment_penalty

        self.reward += pitch_reward
        print(f"  Pitch reward: {pitch_reward}")

    def _calculate_line_of_sight_reward(self):
        # increment reward if chicken in line of sight
        entities_dict = self.data["entities"]
        distance_from_chicken = utils.distance(self.state.get_xz_delta())

        try:
            if self.data["LineOfSight"]["hitType"] == "entity":
                print("LOS hits entity reward: 30")
                self.reward += 30
                # increment reward if chicken is in range
                if self.data["LineOfSight"]["inRange"]:
                    print("LOS hits entity, and agent in range reward: 60")
                    self.reward += 60
            elif utils.distance((self.data["LineOfSight"]["x"] - entities_dict[1]['x'],
                                 self.data["LineOfSight"]["z"] - entities_dict[1]['z'])) <= 1:
                los_distance = utils.distance((self.data["LineOfSight"]["x"] - entities_dict[1]['x'],
                                               self.data["LineOfSight"]["z"] - entities_dict[1]['z']))
                print(f"LOS hits nearby block to chicken reward: {(1 - los_distance) * 15}")
                self.reward += ((1 - los_distance) * 15)
                if distance_from_chicken <= ATTACK_RANGE:
                    in_kill_range_reward = (ATTACK_RANGE - distance_from_chicken) * 6
                    print(
                        f"LOS hits nearby block to chicken reward, and agent close: {in_kill_range_reward}")
                    self.reward += in_kill_range_reward
        except KeyError:
            print("penalized -300 for: LOS doesn't hit anything")
            self.reward -= 300

    def _calculate_damage_dealt_reward(self):
        # reward for damaging target
        damage_dealt = self.data['DamageDealt']
        if damage_dealt > self.current_damage_dealt:
            if self.action == 5:
                self.reward += ((damage_dealt - self.current_damage_dealt) * 10)  # Each half a heart dmg = 10 DamageDealt
                print(f"reward for dealing dmg: {((damage_dealt - self.current_damage_dealt) * 10)}")

    def _calculate_chickens_killed_reward(self):
        killedChicken = False
        inSync = True
        mobs_killed = self.data['MobsKilled']
        if mobs_killed > 0:
            # Check if action was "attack" before rewarding - deal with possible client-server synchronization issue
            if self.action == 5:
                self.reward += (mobs_killed * 5000)
                print(f"reward for KILLING CHICKEN: {(mobs_killed * 5000)}")
            else:
                inSync = False

            killedChicken = True

        return killedChicken, inSync

    def calculate_reward(self):
        penalty_step = self._penalize_for_looking_above_eye_level()
        self._penalize_for_pitching_down_when_already_looking_straight_down()
        # self._penalize_for_attacking()
        # self._penalize_for_getting_stuck()
        self._calculate_distance_from_chicken_reward()
        self._calculate_yaw_from_chicken_reward()
        # self._calculate_pitch_from_chicken_reward()
        self._penalize_near_wall()
        # self._calculate_line_of_sight_reward()
        self._calculate_damage_dealt_reward()
        killedChicken, inSync = self._calculate_chickens_killed_reward()
        return self.reward, killedChicken, inSync, penalty_step
