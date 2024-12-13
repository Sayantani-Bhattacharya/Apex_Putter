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
        Add a box to the planning scene asynchronously.
    + remove_box_async() -
        Remove a box from the planning scene asynchronously.
    + add_sphere_async() -
        Add a sphere to the planning scene asynchronously.
    + add_sphere() -
        Synchronous wrapper for add_sphere_async().
    + remove_box() -
        Synchronous wrapper for remove_box_async().
    + attach_object_async() -
        Attach an object to a link asynchronously.
    + detach_object_async() -
        Detach an object from a link asynchronously.
    + load_scene_from_parameters_async() -
        Load a scene from a list of parameters asynchronously.

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
        Initialize the PlanningSceneClass with an existing ROS 2 node.

        Args:
        ----
        node (Node): The running ROS node used to interface with ROS.

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
        Add a box to the planning scene asynchronously.

        Args:
        ----
        box_id (str): The unique identifier for the box.
        size (tuple[float, float, float]): Dimensions of the box (x, y, z).
        position (tuple[float, float, float]): Position of the box (x, y, z).
        orientation (tuple[float, float, float, float]): Quaternion orientation (x, y, z, w).
        frame_id (str): The reference frame in which the box is defined.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

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

        result = await self._apply_collision_object(collision_object)
        return result

    async def remove_box_async(self, box_id, frame_id='base'):
        """
        Remove a box from the planning scene asynchronously.

        Args:
        ----
        box_id (str): The unique identifier of the box to remove.
        frame_id (str): The reference frame in which the box is defined.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

        """
        collision_object = CollisionObject()
        collision_object.id = box_id
        collision_object.header.frame_id = frame_id
        collision_object.operation = CollisionObject.REMOVE

        result = await self._apply_collision_object(collision_object)
        return result

    async def add_sphere_async(self,
                               sphere_id,
                               radius,
                               position,
                               orientation=(0.0, 0.0, 0.0, 1.0),
                               frame_id='base'):
        """
        Add a sphere to the planning scene asynchronously.

        Args:
        ----
        sphere_id (str): The unique identifier for the sphere.
        radius (float): The radius of the sphere.
        position (tuple[float, float, float]): Position of the sphere (x, y, z).
        orientation (tuple[float, float, float, float]): Quaternion orientation (x, y, z, w).
        frame_id (str): The reference frame in which the sphere is defined.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

        """
        collision_object = CollisionObject()
        collision_object.id = sphere_id
        collision_object.header.frame_id = frame_id

        # Define the sphere shape
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [radius]

        # Define the sphere pose
        sphere_pose = Pose()
        sphere_pose.position.x, sphere_pose.position.y, sphere_pose.position.z = position
        sphere_pose.orientation.x, sphere_pose.orientation.y, \
            sphere_pose.orientation.z, sphere_pose.orientation.w = orientation

        collision_object.primitives = [sphere]
        collision_object.primitive_poses = [sphere_pose]
        collision_object.operation = CollisionObject.ADD

        result = await self._apply_collision_object(collision_object)
        return result

    def add_sphere(self,
                   sphere_id,
                   radius,
                   position,
                   orientation=(0.0, 0.0, 0.0, 1.0),
                   frame_id='base'):
        """
        Add a sphere to the planning scene synchronously.

        Args:
        sphere_id (str): The unique identifier for the sphere.
        radius (float): The radius of the sphere.
        position (tuple[float, float, float]): Position of the sphere (x, y, z).
        orientation (tuple[float, float, float, float]): Quaternion orientation (x, y, z, w).
        frame_id (str): The reference frame in which the sphere is defined.

        Returns
        -------
            Future: A future that resolves to an ApplyPlanningScene.Response.

        """
        executor = rclpy.get_global_executor()

        if executor is None:
            raise RuntimeError('No executor is running. Make sure rclpy.init() has been called')

        self.future = Future()
        executor.create_task(
            self.add_sphere_async(sphere_id, radius, position, orientation, frame_id)
        ).add_done_callback(self.done_callback)

        self.node.get_logger().info('Task done')

        return self.future

    def remove_box(self, box_id, frame_id='base'):
        """
        Remove a box from the planning scene synchronously.

        Args:
        box_id (str): The unique identifier of the box to remove.
        frame_id (str): The reference frame in which the box is defined.

        Returns
        -------
            Future: A future that resolves to an ApplyPlanningScene.Response.

        """
        executor = rclpy.get_global_executor()

        if executor is None:
            raise RuntimeError('No executor is running. Make sure rclpy.init() has been called')

        self.future = Future()
        executor.create_task(self.remove_box_async(box_id,
                                                   frame_id)).add_done_callback(self.done_callback)
        self.node.get_logger().info('Task done')
        return self.future

    async def attach_object_async(self, object_id, link_name):
        """
        Attach an object to a link in the planning scene asynchronously.

        Args:
        object_id (str): The unique identifier of the object to attach.
        link_name (str): The name of the link to which the object will be attached.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

        """
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.ADD

        # Remove the object from the world
        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.operation = CollisionObject.REMOVE

        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.robot_state.attached_collision_objects.append(attached_object)
        planning_scene.world.collision_objects.append(collision_object)

        result = await self._apply_planning_scene(planning_scene)
        return result

    async def detach_object_async(self, object_id, link_name):
        """
        Detach an object from a link in the planning scene asynchronously.

        Args:
        object_id (str): The unique identifier of the object to detach.
        link_name (str): The name of the link from which the object will be detached.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

        """
        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.REMOVE

        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.robot_state.is_diff = True
        planning_scene.robot_state.attached_collision_objects.append(attached_object)

        result = await self._apply_planning_scene(planning_scene)
        return result

    async def load_scene_from_parameters_async(self, parameters):
        """
        Load a planning scene from a list of parameters asynchronously.

        Args:
        parameters (list): A list of dictionaries, each defining a box object with keys:
        'id' (str): The object ID.
        'size' (tuple[float, float, float]): The object's dimensions.
        'position' (tuple[float, float, float]): The object's position (x, y, z).
        'orientation' (tuple[float, float, float, float], optional): orientation.
        'frame_id' (str, optional): The frame in which the object is defined.

        Returns
        -------
            None: This method does not return a value.

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
        """
        Apply a given collision object to the planning scene.

        Args:
        collision_object (CollisionObject): The collision object to apply.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the planning scene service.

        """
        planning_scene = PlanningMsg()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)
        result = await self._apply_planning_scene(planning_scene)
        return result

    async def _apply_planning_scene(self, planning_scene):
        """
        Apply the given planning scene changes.

        Args:
        planning_scene (PlanningMsg): The planning scene configuration to apply.

        Returns
        -------
            ApplyPlanningScene.Response: The response from the ApplyPlanningScene service.

        """
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        result = await self.apply_planning_scene_client.call_async(request)
        return result
