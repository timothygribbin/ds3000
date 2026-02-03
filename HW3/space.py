#!/usr/bin/env python
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3, LineSegs, NodePath, TextNode
from panda3d.core import LVector3, LPoint3
from panda3d.core import WindowProperties
from direct.task import Task
from panda3d.core import GeomVertexReader, Filename
import imageio
import math
import numpy as np

root = None

def tuple_vector_to_numpy(tup): 
	return np.array(tup).reshape(-1, 1)

def ensure_tuple(obj):
	"""Check if the input is a tuple. If it's a NumPy array, convert it to a tuple."""
	if isinstance(obj, tuple):
		return obj  # Already a tuple, return as is
	elif isinstance(obj, list):
		return tuple(obj)
	elif isinstance(obj, np.ndarray):
		return tuple(obj.flatten())  # Convert NumPy array to a tuple
	else:
		raise TypeError("Input must be a tuple or a NumPy array.")


class point_cloud():
	def __init__(self, X=None, mean=[1, 1, 1], n = 20, color=(0.6, 0.6, 1, 0.3)):
		if X is None:
			μ = np.array(mean)	
			Σ = np.array([	[1, 0, 0.3],
							[0, 2, 0],
							[0.3, 0, 0.5]])
			X = np.random.multivariate_normal(μ, Σ, n)

		self.X = X
		self.n = self.X.shape[0]
		self.draw_points(self.X)

	def draw_points(self, new_points):
		global root

		self.points = []
		for x in new_points:
			point = root.loader.loadModel("models/misc/sphere")  # Use "models/misc/sphere" for a pure sphere
			point.reparentTo(root.render)
			point.setScale(0.07)
			point.setColor(0.6, 0.6, 1, 0.3)
			point.setPos(x[0], x[1], x[2])

			self.points.append(point)

		self.X = new_points

	def pos(self):
		return self.X


	def delete(self):
		for point in self.points:
			point.removeNode()

	def redraw(self, pos):
		self.delete()
		self.draw_points(pos)




class point():
	def __init__(self, x):
		self.x = ensure_tuple(x)
		self.draw_point(x)

	def draw_point(self, x):
		global root
		self.point = root.loader.loadModel("models/misc/sphere")  # Use "models/misc/sphere" for a pure sphere
		self.point.reparentTo(root.render)
		self.point.setScale(0.07)
		self.point.setColor(0.6, 0.6, 1, 0.3)
		self.point.setPos(x[0], x[1], x[2])

		self.x = tuple_vector_to_numpy(x)

	def pos(self):
		return self.x

	def delete(self):
		self.point.removeNode()

	def redraw(self, new_pos):
		self.delete()
		self.draw_point(new_pos)



	def __rmatmul__(self, other):
		"""Implements other @ self"""
		if isinstance(other, np.ndarray):
			return other @ self.pos  # Right multiplication
		else:
			raise TypeError(f"Unsupported type {type(other)} for matrix multiplication")

	def __matmul__(self, other):
		"""Implements self @ other"""
		print(other)
		print(self.pos)
		print('\n\n\n')

		if isinstance(other, np.ndarray):
			return self.pos.T @ other  # Matrix multiplication
		elif isinstance(other, np.ndarray):
			return self.pos.T @ self.pos  # MyMatrix @ MyMatrix
		else:
			raise TypeError(f"Unsupported type {type(other)} for matrix multiplication")



class vector():
	def __init__(self, render, X, start=(0,0,0), color=(0.7, 0.7, 0, 1), thickness=3):
		self.X = ensure_tuple(X)
		self.render = render
		self.thickness = thickness
		self.draw_line(X, start, color)

	def draw_line(self, pos, start=(0,0,0), color=(0.7, 0.7, 0, 1)):
		self.lines = LineSegs()
		self.lines.setColor(*color)  # Set the line color
		self.lines.setThickness(4)  # Set line thickness
		self.lines.moveTo(*start)  # Start point
		self.lines.drawTo(*pos)  # End point
		self.lines.setThickness(self.thickness)

		self.line_node = NodePath(self.lines.create())
		self.line_node.reparentTo(render)

		self.start = tuple_vector_to_numpy(start)
		self.pos = tuple_vector_to_numpy(pos)

	def pos(self):
		return self.pos

	def delete(self):
		self.line_node.removeNode()

	def redraw(self, pos, start=(0,0,0), color=(0.7, 0.7, 0, 1)):
		pos = ensure_tuple(pos)
		self.delete()
		self.draw_line(pos)



	def __rmatmul__(self, other):
		"""Implements other @ self"""
		if isinstance(other, np.ndarray):
			return other @ self.pos  # Right multiplication
		else:
			raise TypeError(f"Unsupported type {type(other)} for matrix multiplication")

	def __matmul__(self, other):
		"""Implements self @ other"""
		print(other)
		print(self.pos)
		print('\n\n\n')

		if isinstance(other, np.ndarray):
			return self.pos.T @ other  # Matrix multiplication
		elif isinstance(other, np.ndarray):
			return self.pos.T @ self.pos  # MyMatrix @ MyMatrix
		else:
			raise TypeError(f"Unsupported type {type(other)} for matrix multiplication")


class space(ShowBase):
	def __init__(self):
		ShowBase.__init__(self)
		global root
		root = self
		self.is_fullscreen = False

		# Disable default camera controls
		self.disableMouse()

		# Create axes
		self.create_axes()

		# Create grid floor
		self.create_grid()

		# Update camera task
		self.taskMgr.add(self.update_camera, "UpdateCameraTask")


		# Set initial camera position in spherical coordinates
		self.cam_radius = 20
		self.cam_theta = math.pi / 4
		self.cam_phi = math.pi / 4
	

		self.recording = False
		# Mouse movement tracking
		self.accept("mouse1", self.start_mouse_tracking)
		self.accept("mouse1-up", self.stop_mouse_tracking)
		self.accept("f", self.toggle_fullscreen)
		self.accept("r", self.start_recording)
		self.accept("f9", self.capture_screenshot)
		self.taskMgr.add(self.track_mouse, "TrackMouseTask")
		self.mouse_tracking = False
		self.last_mouse_pos = (0, 0)

		# Mouse scroll for zooming
		self.accept("wheel_up", self.zoom_in)
		self.accept("wheel_down", self.zoom_out)

	def load_mesh(self, path):
		mesh = self.loader.load_model(Filename.from_os_specific(path))
		if not mesh:
			print("Failed to load model:", model_path)
			return None

		mesh.reparentTo(self.render)
		return mesh




	def start_recording(self):
		if not self.recording:
			self.recording = True
			self.record_task = self.taskMgr.add(self.record_screen, "RecordScreen")
			print("Recording started. Press 'r' to stop.")
		else:
			self.stop_recording()

	def stop_recording(self):
		self.recording = False
		self.taskMgr.remove(self.record_task)
		self.save_gif()
		print("Recording stopped.")

	def record_screen(self, task):
		if self.recording:
			frame_count = task.frame
			if frame_count % 10 == 0:  # Record every 5th frame
				self.win.saveScreenshot("screenshot.png")
				if not hasattr(self, "frames"):
					self.frames = []
				self.frames.append(imageio.imread("screenshot.png"))
				if len(self.frames) >= 150:  # Adjusted to account for every 5th frame
					self.stop_recording()
					self.save_gif()
			return Task.cont


	def draw_new_basis(self, V, 
						  length=5.0,
						  colors=((1, 1, 0, 1),   # yellow   for v1
								  (1, 0, 1, 1),   # magenta  for v2
								  (0, 1, 1, 1))): # cyan	 for v3
		
		# Draw and label --------------------------------------------
		vec1 = self.create_vector(V[0,:], color=colors[0], thickness=5)
		vec2 = self.create_vector(V[1,:], color=colors[1], thickness=5)
		vec3 = self.create_vector(V[2,:], color=colors[2], thickness=5)
	
		# Optional tip labels (uncomment if you want them right away)
		self.create_axis_label("x", * (1.1 * V[0,:]), colors[0])
		self.create_axis_label("y", * (1.1 * V[1,:]), colors[1])
		self.create_axis_label("z", * (1.1 * V[2,:]), colors[2])
	
		return V



	def create_vector(self, pos, start=(0,0,0), color=(0.7, 0.7, 0, 1), thickness=3):
		return vector(self.render, pos, start=start, color=color, thickness=thickness)


	def create_grid(self, size=10, spacing=1):
		grid = LineSegs()
		grid.setColor(0.5, 0.5, 0.5, 1)
		
		for i in range(-size, size + 1):
			grid.moveTo(i * spacing, -size * spacing, 0)
			grid.drawTo(i * spacing, size * spacing, 0)
			grid.moveTo(-size * spacing, i * spacing, 0)
			grid.drawTo(size * spacing, i * spacing, 0)
		
		grid_node = grid.create()
		self.render.attachNewNode(grid_node)



	def create_axes(self, length=5):
		axes = LineSegs()
		axes.setThickness(4.0)  # Make axes thicker
		
		# X Axis (Red)
		axes.setColor(0, 0, 0, 1)
		axes.moveTo(0, 0, 0)
		axes.drawTo(length, 0, 0)
		
		# Y Axis (Green)
		axes.setColor(0, 0, 0, 1)
		axes.moveTo(0, 0, 0)
		axes.drawTo(0, length, 0)
		
		# Z Axis (Blue)
		axes.setColor(0, 0, 0, 1)
		axes.moveTo(0, 0, 0)
		axes.drawTo(0, 0, length)
		
		axes_node = axes.create()
		axes_np = self.render.attachNewNode(axes_node)

		# Add labels
		self.create_axis_label("X", length + 0.3, 0, 0, (0, 0, 0, 1))
		self.create_axis_label("Y", 0, length + 0.3, 0, (0, 0, 0, 1))
		self.create_axis_label("Z", 0, 0, length + 0.3, (0, 0, 0, 1))
	




	def create_axis_label(self, text, x, y, z, color):
		label = TextNode(text)
		label.setText(text)
		label.setTextColor(*color)
		label.setAlign(TextNode.ACenter)
		label_node = self.render.attachNewNode(label)
		label_node.setScale(0.5)
		label_node.setPos(x, y, z)
		label_node.setBillboardPointEye()  # Ensures visibility from both sides

	def update_camera(self, task):
		x = self.cam_radius * math.sin(self.cam_phi) * math.cos(self.cam_theta)
		y = self.cam_radius * math.sin(self.cam_phi) * math.sin(self.cam_theta)
		z = self.cam_radius * math.cos(self.cam_phi)
		self.camera.setPos(x, y, z)
		self.camera.lookAt(0, 0, 1)
		return Task.cont

	def start_mouse_tracking(self):
		self.mouse_tracking = True
		if self.mouseWatcherNode.hasMouse():
			self.last_mouse_pos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())

	def stop_mouse_tracking(self):
		self.mouse_tracking = False

	def track_mouse(self, task):
		if self.mouse_tracking and self.mouseWatcherNode.hasMouse():
			new_x, new_y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
			dx = new_x - self.last_mouse_pos[0]
			dy = new_y - self.last_mouse_pos[1]
			
			self.cam_theta -= dx * 2
			self.cam_phi = max(0.1, min(math.pi - 0.1, self.cam_phi + dy * 2))  # Corrected inversion
			
			self.last_mouse_pos = (new_x, new_y)
		return Task.cont

	def zoom_in(self):
		self.cam_radius = max(2, self.cam_radius - 1)

	def zoom_out(self):
		self.cam_radius += 1

	def toggle_fullscreen(self):
		"""Toggle between fullscreen and windowed mode."""
		self.is_fullscreen = not self.is_fullscreen  # Flip state
		
		props = WindowProperties()
		props.setFullscreen(self.is_fullscreen)
		base.win.requestProperties(props)  # Apply new properties

	def save_gif(self):
		imageio.mimsave("record.gif", self.frames, fps=6)
		print("GIF saved.")

	def capture_screenshot(self):
		filename = "screenshot.png"
		self.win.saveScreenshot(filename)
		print(f"Screenshot saved to {filename}")
