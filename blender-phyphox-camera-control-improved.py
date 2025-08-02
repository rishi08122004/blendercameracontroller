import bpy
import csv
import math
import os
from mathutils import Vector, Quaternion

class PhyphoxCameraController:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.sensor_data = []
        self.camera = None
        self.initial_camera_loc = Vector((0, 0, 0))
        self.initial_camera_rot = Vector((0, 0, 0))
        
    def load_data(self):
        """Load data from the CSV file exported from Phyphox."""
        try:
            with open(self.csv_file_path, 'r') as file:
                # Skip header lines if they exist (Phyphox usually includes headers)
                reader = csv.reader(file)
                
                # Skip metadata lines 
                line_count = 0
                for row in reader:
                    line_count += 1
                    # Usually Phyphox has a header row with column names
                    if line_count > 1 and len(row) >= 4:  # Time, x, y, z format
                        try:
                            time = float(row[0])
                            x = float(row[1])
                            y = float(row[2])
                            z = float(row[3])
                            self.sensor_data.append({'time': time, 'x': x, 'y': y, 'z': z})
                        except ValueError:
                            # Skip rows that can't be converted to float
                            continue
            
            print(f"Loaded {len(self.sensor_data)} data points from {self.csv_file_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_camera(self):
        """Get or create a camera in the Blender scene."""
        # Use existing camera or create a new one
        if 'Camera' in bpy.data.objects:
            self.camera = bpy.data.objects['Camera']
        else:
            camera_data = bpy.data.cameras.new('Camera')
            self.camera = bpy.data.objects.new('Camera', camera_data)
            bpy.context.collection.objects.link(self.camera)
        
        # Save initial camera position and rotation
        self.initial_camera_loc = self.camera.location.copy()
        self.initial_camera_rot = self.camera.rotation_euler.copy()
        
        # Make sure the camera is the active camera
        bpy.context.scene.camera = self.camera
        return self.camera
    
    def apply_low_pass_filter(self, data, cutoff=0.1):
        """Apply a simple low-pass filter to the data.
        cutoff is between 0 and 1, where lower values mean more filtering"""
        filtered_data = []
        prev_x, prev_y, prev_z = data[0]['x'], data[0]['y'], data[0]['z']
        
        for point in data:
            # Apply low-pass filter: new_value = prev_value + cutoff * (current_raw - prev_value)
            filtered_x = prev_x + cutoff * (point['x'] - prev_x)
            filtered_y = prev_y + cutoff * (point['y'] - prev_y)
            filtered_z = prev_z + cutoff * (point['z'] - prev_z)
            
            filtered_data.append({
                'time': point['time'],
                'x': filtered_x,
                'y': filtered_y,
                'z': filtered_z
            })
            
            prev_x, prev_y, prev_z = filtered_x, filtered_y, filtered_z
            
        return filtered_data
    
    def apply_kalman_filter(self, data, process_noise=0.01, measurement_noise=0.1):
        """Apply a simplified Kalman filter for smoother tracking."""
        filtered_data = []
        
        # Initial state
        x_est, y_est, z_est = data[0]['x'], data[0]['y'], data[0]['z']
        p_x, p_y, p_z = 1.0, 1.0, 1.0  # Initial uncertainty
        
        for point in data:
            # Prediction (no motion model, so just use previous estimate)
            p_x += process_noise
            p_y += process_noise
            p_z += process_noise
            
            # Update
            k_x = p_x / (p_x + measurement_noise)
            k_y = p_y / (p_y + measurement_noise)
            k_z = p_z / (p_z + measurement_noise)
            
            x_est = x_est + k_x * (point['x'] - x_est)
            y_est = y_est + k_y * (point['y'] - y_est)
            z_est = z_est + k_z * (point['z'] - z_est)
            
            p_x = (1 - k_x) * p_x
            p_y = (1 - k_y) * p_y
            p_z = (1 - k_z) * p_z
            
            filtered_data.append({
                'time': point['time'],
                'x': x_est,
                'y': y_est,
                'z': z_est
            })
            
        return filtered_data
    
    def downsample_data(self, data, factor=5):
        """Downsample the data by keeping only every nth point."""
        return data[::factor]
    
    def smooth_data(self, data, window_size=5):
        """Apply a simple moving average smoothing to the data."""
        if window_size < 2:
            return data
            
        smoothed_data = []
        for i in range(len(data)):
            # Calculate window boundaries
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            
            # Calculate averages
            x_sum = sum(point['x'] for point in window)
            y_sum = sum(point['y'] for point in window)
            z_sum = sum(point['z'] for point in window)
            
            # Create smoothed data point
            smoothed_data.append({
                'time': data[i]['time'],
                'x': x_sum / len(window),
                'y': y_sum / len(window),
                'z': z_sum / len(window)
            })
            
        return smoothed_data
    
    def remove_outliers(self, data, threshold=2.0):
        """Remove outliers that are more than threshold standard deviations from the mean."""
        if len(data) < 4:
            return data
            
        # Calculate means and standard deviations
        x_values = [point['x'] for point in data]
        y_values = [point['y'] for point in data]
        z_values = [point['z'] for point in data]
        
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        z_mean = sum(z_values) / len(z_values)
        
        x_std = math.sqrt(sum((x - x_mean) ** 2 for x in x_values) / len(x_values))
        y_std = math.sqrt(sum((y - y_mean) ** 2 for y in y_values) / len(y_values))
        z_std = math.sqrt(sum((z - z_mean) ** 2 for z in z_values) / len(z_values))
        
        # Remove outliers
        filtered_data = []
        for point in data:
            x_diff = abs(point['x'] - x_mean)
            y_diff = abs(point['y'] - y_mean)
            z_diff = abs(point['z'] - z_mean)
            
            if (x_diff < threshold * x_std and 
                y_diff < threshold * y_std and 
                z_diff < threshold * z_std):
                filtered_data.append(point)
        
        return filtered_data if filtered_data else data  # Return original if all removed
    
    def animate_camera(self, scale_factor=0.5, smoothing=5, filter_type='kalman', 
                       use_rotation=True, damping=0.8, downsample=2):
        """Animate the camera based on loaded sensor data with improved filtering."""
        if not self.sensor_data:
            print("No data to animate. Load data first.")
            return False
        
        if not self.camera:
            self.setup_camera()
        
        # Clear existing animation data
        self.camera.animation_data_clear()
        
        # Process data with various filtering techniques
        processed_data = self.sensor_data
        
        # Remove extreme outliers first
        processed_data = self.remove_outliers(processed_data, threshold=3.0)
        
        # Downsample to reduce jitter
        if downsample > 1:
            processed_data = self.downsample_data(processed_data, factor=downsample)
        
        # Apply selected filter
        if filter_type == 'moving_average':
            processed_data = self.smooth_data(processed_data, window_size=smoothing)
        elif filter_type == 'low_pass':
            processed_data = self.apply_low_pass_filter(processed_data, cutoff=0.1)
        elif filter_type == 'kalman':
            processed_data = self.apply_kalman_filter(processed_data)
        
        # Apply final smoothing
        processed_data = self.smooth_data(processed_data, window_size=smoothing)
        
        # Calculate frame rate and total frames
        fps = bpy.context.scene.render.fps
        start_time = processed_data[0]['time']
        end_time = processed_data[-1]['time']
        duration = end_time - start_time
        total_frames = int(duration * fps)
        
        # Set scene frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = total_frames
        
        # Initial position (possibly after calibration)
        calibration_samples = min(20, len(processed_data) // 10)  # Use 10% of data or 20 samples
        initial_x = sum(data['x'] for data in processed_data[:calibration_samples]) / calibration_samples
        initial_y = sum(data['y'] for data in processed_data[:calibration_samples]) / calibration_samples
        initial_z = sum(data['z'] for data in processed_data[:calibration_samples]) / calibration_samples
        
        # Previous values for damping
        prev_loc = Vector(self.initial_camera_loc)
        prev_rot = Vector(self.initial_camera_rot)
        
        # Set keyframes for each data point
        for i, data_point in enumerate(processed_data):
            # Calculate frame number
            time_from_start = data_point['time'] - start_time
            frame = int(time_from_start * fps) + 1
            
            # Position offset from initial reading - apply scale and damping
            x_offset = (data_point['x'] - initial_x) * scale_factor
            y_offset = (data_point['y'] - initial_y) * scale_factor
            z_offset = (data_point['z'] - initial_z) * scale_factor
            
            # Calculate new location with damping
            new_loc = Vector((
                self.initial_camera_loc.x + x_offset,
                self.initial_camera_loc.y + y_offset,
                self.initial_camera_loc.z + z_offset
            ))
            
            # Apply damping (interpolation between previous and new)
            damped_loc = prev_loc.lerp(new_loc, 1.0 - damping)
            
            # Set camera position
            self.camera.location = damped_loc
            self.camera.keyframe_insert(data_path="location", frame=frame)
            prev_loc = damped_loc
            
            # Optional: Set camera rotation based on sensor data
            if use_rotation:
                # Get a gentler rotation from sensor data
                # Reduce rotation amount significantly to avoid dizziness
                rotation_scale = scale_factor * 0.1
                
                # Simple rotation from acceleration vector
                direction = Vector((x_offset, y_offset, z_offset)).normalized()
                up = Vector((0, 0, 1))
                
                # Only rotate if we have significant movement
                if direction.length > 0.01:
                    # Create a rotation that points the camera in the direction of movement
                    # but keeps it upright (this is a simplification)
                    rot_z = math.atan2(direction.y, direction.x)
                    rot_y = math.asin(direction.z)
                    rot_x = 0  # Keep camera level with horizon
                    
                    new_rot = Vector((rot_x, rot_y, rot_z)) * rotation_scale
                    damped_rot = prev_rot.lerp(new_rot, 1.0 - damping)
                    
                    self.camera.rotation_euler = (
                        self.initial_camera_rot.x + damped_rot.x,
                        self.initial_camera_rot.y + damped_rot.y,
                        self.initial_camera_rot.z + damped_rot.z
                    )
                    self.camera.keyframe_insert(data_path="rotation_euler", frame=frame)
                    prev_rot = damped_rot
        
        # Set interpolation to bezier for smoother transitions
        for fcurve in self.camera.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'BEZIER'
                kf.handle_left_type = 'AUTO'
                kf.handle_right_type = 'AUTO'
        
        print(f"Animation created with {len(processed_data)} keyframes over {total_frames} frames")
        return True

# --------------------------------------
# User Interface for the script
# --------------------------------------

class PHYPHOX_PT_panel(bpy.types.Panel):
    """Panel to control Phyphox camera animation"""
    bl_label = "Phyphox Camera Control"
    bl_idname = "PHYPHOX_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Phyphox'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # File selection
        layout.label(text="CSV File:")
        row = layout.row()
        row.prop(scene, "phyphox_csv_path", text="")
        
        # Parameters
        layout.label(text="Animation Parameters:")
        
        row = layout.row()
        row.prop(scene, "phyphox_scale_factor", text="Scale Factor")
        
        row = layout.row()
        row.prop(scene, "phyphox_smoothing", text="Smoothing")
        
        row = layout.row()
        row.prop(scene, "phyphox_filter_type", text="Filter Type")
        
        row = layout.row()
        row.prop(scene, "phyphox_damping", text="Damping")
        
        row = layout.row()
        row.prop(scene, "phyphox_downsample", text="Downsample")
        
        row = layout.row()
        row.prop(scene, "phyphox_use_rotation", text="Use Rotation")
        
        # Generate Animation Button
        row = layout.row()
        row.operator("phyphox.generate_animation", text="Generate Animation")


class PHYPHOX_OT_generate_animation(bpy.types.Operator):
    """Generate camera animation from Phyphox data"""
    bl_idname = "phyphox.generate_animation"
    bl_label = "Generate Camera Animation"
    
    def execute(self, context):
        scene = context.scene
        
        if not scene.phyphox_csv_path or not os.path.exists(bpy.path.abspath(scene.phyphox_csv_path)):
            self.report({'ERROR'}, "Invalid CSV file path")
            return {'CANCELLED'}
        
        controller = PhyphoxCameraController(bpy.path.abspath(scene.phyphox_csv_path))
        if not controller.load_data():
            self.report({'ERROR'}, "Failed to load data from CSV")
            return {'CANCELLED'}
        
        success = controller.animate_camera(
            scale_factor=scene.phyphox_scale_factor,
            smoothing=scene.phyphox_smoothing,
            filter_type=scene.phyphox_filter_type,
            use_rotation=scene.phyphox_use_rotation,
            damping=scene.phyphox_damping,
            downsample=scene.phyphox_downsample
        )
        
        if success:
            self.report({'INFO'}, "Camera animation created successfully")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to create animation")
            return {'CANCELLED'}


# Register the addon
def register():
    # Register properties
    bpy.types.Scene.phyphox_csv_path = bpy.props.StringProperty(
        name="CSV File",
        description="Path to the Phyphox exported CSV file",
        default="",
        subtype='FILE_PATH'
    )
    
    bpy.types.Scene.phyphox_scale_factor = bpy.props.FloatProperty(
        name="Scale Factor",
        description="Scale factor for camera movement (lower = less movement)",
        default=0.5,
        min=0.01,
        max=10.0
    )
    
    bpy.types.Scene.phyphox_smoothing = bpy.props.IntProperty(
        name="Smoothing",
        description="Smoothing window size (higher = smoother but less responsive)",
        default=5,
        min=1,
        max=50
    )
    
    bpy.types.Scene.phyphox_filter_type = bpy.props.EnumProperty(
        name="Filter Type",
        description="Type of filter to apply to the data",
        items=[
            ('moving_average', "Moving Average", "Simple moving average filter"),
            ('low_pass', "Low Pass", "Low pass filter (reduces high frequency noise)"),
            ('kalman', "Kalman", "Kalman filter (best for tracking)")
        ],
        default='kalman'
    )
    
    bpy.types.Scene.phyphox_use_rotation = bpy.props.BoolProperty(
        name="Use Rotation",
        description="Animate camera rotation (can be disorienting)",
        default=False
    )
    
    bpy.types.Scene.phyphox_damping = bpy.props.FloatProperty(
        name="Damping",
        description="Damping factor (higher = smoother but more lag)",
        default=0.8,
        min=0.0,
        max=0.99
    )
    
    bpy.types.Scene.phyphox_downsample = bpy.props.IntProperty(
        name="Downsample",
        description="Only use every nth data point (reduces jitter)",
        default=2,
        min=1,
        max=10
    )
    
    # Register classes
    bpy.utils.register_class(PHYPHOX_PT_panel)
    bpy.utils.register_class(PHYPHOX_OT_generate_animation)


def unregister():
    # Unregister properties
    del bpy.types.Scene.phyphox_csv_path
    del bpy.types.Scene.phyphox_scale_factor
    del bpy.types.Scene.phyphox_smoothing
    del bpy.types.Scene.phyphox_filter_type
    del bpy.types.Scene.phyphox_use_rotation
    del bpy.types.Scene.phyphox_damping
    del bpy.types.Scene.phyphox_downsample
    
    # Unregister classes
    bpy.utils.unregister_class(PHYPHOX_PT_panel)
    bpy.utils.unregister_class(PHYPHOX_OT_generate_animation)


# For testing in Blender's text editor
if __name__ == "__main__":
    register()
