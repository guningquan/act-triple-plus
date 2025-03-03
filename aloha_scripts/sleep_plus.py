from interbotix_xs_modules.arm import InterbotixManipulatorXS
try:
    from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers, torque_off
except ImportError:
    from robot_utils import move_arms, torque_on, move_grippers,torque_off
import rospy
import threading,time
import argparse

def move_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    # torque_off(robot)

def move_shut_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    time.sleep(1)
    torque_off(robot)

def shut_down_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.76, 1.55, 0, 0.0, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    MASTER_GRIPPER_JOINT_OPEN = 0.02417

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_puppet_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.76, 1.55, 0, 0.0, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    MASTER_GRIPPER_JOINT_OPEN = 0.02417

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_puppet_robots():
    # rospy.init_node('sleep_plus')

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910

    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def sleep_master_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.76, 1.55, 0, 0.0, 0)
    MASTER_GRIPPER_JOINT_OPEN = 0.02417

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()





def main():
    parser = argparse.ArgumentParser(description="Control script for robots")
    parser.add_argument("--sleep", action="store_true",
                        help="Put all robots into sleep mode")
    parser.add_argument("--shut_down", action="store_true",
                        help="Shut down all robots")
    parser.add_argument("--shut_down_puppet", action="store_true",
                        help="Shut down puppet robots")

    args = parser.parse_args()

    if args.sleep:
        sleep_all_robots()
    elif args.shut_down:
        shut_down_all_robots()
    elif args.shut_down_puppet:
        shut_down_puppet_robots()
    else:
        print("Please provide a valid argument (--sleep or --shut_down).")

if __name__ == "__main__":
    rospy.init_node('sleep_plus')
    main()
