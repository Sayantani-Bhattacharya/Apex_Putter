"""
PlanningSceneClass maintains the planning scene.

The PlanningSceneClass allows for the simulation environment
in Rviz to be manipulated by adding simulated objects that
are considered during motion planning.

Clients
-------
    + /apply_planning_scene (ApplyPlanningScene) - Service client
        for applying changes to the planning scene

Methods
-------
    + add_box_async() -
        Add box to planning scene.
    + remove_box_async() -
        Remove box from planning scene.
    + add_sphere_async() -
        Add sphere to planning scene.
    + add_sphere() -
        Synchronous wrapper for add_sphere_async()
    + remove_box() -
        Synchronous wrapper for remove_box_async()
    + attach_box_async() -
        Attach a box to a link.
    + detach_box_async() -
        Detach a box from a link.
    + load_scene_from_parameters_async() -
        Load a scene from list of parameters.

"""

from asyncio import Future

from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from moveit_msgs.msg import PlanningScene as PlanningMsg
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class PlanningSceneClass:
    def __init__(self, node: Node):
        """
        Initialize the PlanningScene with an existing ROS 2 node.

        Args:
            node: The running ROS node used to interface with ROS.
        """
        self.node = node
        self.apply_planning_scene_client = self.node.create_client(
            ApplyPlanningScene, '/apply_planning_scene',
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Wait for the service to become available
        self.node.get_logger().info('Waiting for /apply_planning_scene service...')
        while not self.apply_planning_scene_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('Service /apply_planning_scene not available.')
            # raise RuntimeError('Service /apply_planning_scene not available.')

        self.node.get_logger().info('Service /apply_planning_scene is now available.')

        # Create the joint state subscriber
        self.joint_state_subscriber = None

    def done_callback(self, task):
        result = task.result()
        self.future.set_result(result)

    async def add_box_async(self,
                            box_id,
                            size,
                            position,
                            orientation=(0.0, 0.0, 0.0, 1.0),
                            frame_id='base'):
        """
        Add a box in the planning scene.

        Args
        ----
        box_id (string): id of the box
        size (tuple): dimensions of the box (x, y, z)
        position (tuple): position of box (x, y, z)
        orientation (Quaternion): orientaion of the box
        frame_id (string): frame in which the box is published

        """
        collision_object = CollisionObject()
        collision_object.id = box_id
        collision_object.header.frame_id = frame_id

        # Define the box shape
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)

        # Define the box pose
        box_pose = Pose()
        box_pose.position.x, box_pose.position.y, box_pose.position.z = position
        box_pose.orientation.x, box_pose.orientation.y, \
            box_pose.orientation.z, box_pose.orientation.w = orientation

        collision_object.primitives = [box]
        collision_object.primitive_poses = [box_pose]
        collision_object.operation = CollisionObject.ADD

        # Apply the collision object to the planning scene
        result = await self._apply_collision_object(collision_object)
        return result

    async def remove_box_async(self, box_id, frame_id='base'):
        """
        Remove box from the planning scene asynchronously.

        Args
        ----
        box_id (string): id of the box
        frame_id (string): frame in which the box is published

        """
        collision_object = CollisionObject()
        collision_object.id = box_id
        collision_object.header.frame_id = frame_id
        collision_object.operation = CollisionObject.REMOVE

        # Apply the collision object to the planning scene
        result = await self._apply_collision_object(collision_object)
        return result

    async def add_sphere_async(self,
                               sphere_id,
                               radius,
                               position,
                               orientation=(0.0, 0.0, 0.0, 1.0),
                               frame_id='base'):
        """
        Add a sphere in the planning scene async.

        Args
        ----
        sphere_id (string): id of the sphere
        radius (radius): radius of the sphere
        position (tuple): position of sphere
        orientation (Quaternion): orientaion of the sphere
        frame_id (string): frame in which the sphere is published

        """
        collision_object = CollisionObject()
        collision_object.id = sphere_id
        collision_object.header.frame_id = frame_id

        # Define the sphere shape
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        # For a sphere, dimensions[0] is the radius
        sphere.dimensions = [radius]

        # Define the sphere pose
        sphere_pose = Pose()
        sphere_pose.position.x, sphere_pose.position.y, sphere_pose.position.z = position
        sphere_pose.orientation.x, sphere_pose.orientation.y, sphere_pose.orientation.z, \
            sphere_pose.orientation.w = orientation

        collision_object.primitives = [sphere]
        collision_object.primitive_poses = [sphere_pose]
        collision_object.operation = CollisionObject.ADD

        # Apply the collision object to the planning scene
        result = await self._apply_collision_object(collision_object)
        return result

    def add_sphere(self,
                   sphere_id,
                   radius,
                   position,
                   orientation=(0.0, 0.0, 0.0, 1.0),
                   frame_id='base'):
        """
        Add a sphere in the planning scene.

        Args
        ----
        sphere_id (string): id of the sphere
        radius (radius): radius of the sphere
        position (tuple): position of sphere
        orientation (Quaternion): orientaion of the sphere
        frame_id (string): frame in which the sphere is published

        """
        executor = rclpy.get_global_executor()

        if executor is None:
            raise RuntimeError(
                'No executor is running. Make sure rclpy.init() has been called')

        # Create a new future for this request
        self.future = Future()

        executor.create_task(
            self.add_sphere_async(
                sphere_id,
                radius,
                position,
                orientation,
                frame_id
            )
        ).add_done_callback(self.done_callback)

        self.node.get_logger().info('Task done')

        return self.future

    def remove_box(self, box_id, frame_id='base'):
        """
        Remove box from the planning scene.

        Args
        ----
        box_id (string): id of the box
        frame_id (string): frame in which the box is published

        """
        executor = rclpy.get_global_executor()

        if executor is None:
            raise RuntimeError(
                'No executor is running. Make sure rclpy.init() has been called')

        # Create a new future for this request
        self.future = Future()

        executor.create_task(
            self.remove_box_async(
                box_id,
                frame_id
            )
        ).add_done_callback(self.done_callback)

        self.node.get_logger().info('Task done')

        return self.future

    async def attach_object_async(self, object_id, link_name):
        """
        Attach an object to a link in the planning scene.

        Args
        ----
        object_id (string): id of the object
        link_name (string): name of the link

        """
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.ADD

        # Remove the object from the world collision objects
        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.operation = CollisionObject.REMOVE

        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.robot_state.attached_collision_objects.append(
            attached_object)
        planning_scene.world.collision_objects.append(collision_object)

        result = await self._apply_planning_scene(planning_scene)
        return result

    async def detach_object_async(self, object_id, link_name):
        """
        Detach an object from a link in the planning scene.

        Args
        ----
        object_id (string): id of the object
        link_name (string): name of the link

        """
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.REMOVE

        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.robot_state.is_diff = True
        planning_scene.robot_state.attached_collision_objects.append(
            attached_object)

        result = await self._apply_planning_scene(planning_scene)
        return result

    async def load_scene_from_parameters_async(self, parameters):
        """
        Load a planning scene from parameters.

        Args
        ----
        parameters (list): list of planning scene objects

        """
        for param in parameters:
            await self.add_box_async(
                box_id=param['id'],
                size=param['size'],
                position=param['position'],
                orientation=param.get('orientation', (0.0, 0.0, 0.0, 1.0)),
                frame_id=param.get('frame_id', 'base')
            )

    async def _apply_collision_object(self, collision_object):
        """Apply collision object to planning scene."""
        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)
        result = await self._apply_planning_scene(planning_scene)
        return result

    async def _apply_planning_scene(self, planning_scene):
        """Apply planning scene changes."""
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        result = await self.apply_planning_scene_client.call_async(request)
        return result
