import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LiDARDataParser:
    def __init__(self, serial_port='/dev/ttyAMA0', baud_rate=230400):
        """
        Initialize the LiDAR data parser with visualization
        
        :param serial_port: Serial port for LiDAR communication
        :param baud_rate: Communication baud rate
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', alpha=0.7)
        self.ax.set_xlim(-5000, 5000)  # Adjust based on expected LiDAR range (in mm)
        self.ax.set_ylim(-5000, 5000)
        self.ax.set_title('LiDAR Point Cloud')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.grid(True)
        
        # Data storage for visualization
        self.x_data = []
        self.y_data = []
        self.confidence_data = []
        
    def crc_check(self, data):
        """
        Perform CRC verification on the data packet
        
        :param data: Full data packet
        :return: True if CRC check passes, False otherwise
        """
        # Simple CRC calculation (you may need to adjust based on specific LiDAR model)
        crc = 0
        for byte in data[:-1]:  # Exclude the last byte (CRC itself)
            crc ^= byte
        return crc == data[-1]
    
    def parse_packet(self, packet):
        """
        Parse a complete LiDAR data packet
        
        :param packet: Raw data packet
        :return: Parsed point cloud data
        """
        # Verify packet header and structure
        if len(packet) < 10:  # Minimum packet size
            return None
        
        # Check header
        if packet[0] != 0x54:
            return None
        
        # Verify packet type and point count
        if packet[1] != 0x2C:
            return None
        
        # Extract packet information
        speed = struct.unpack('<H', packet[2:4])[0]  # Little-endian 2-byte unsigned short
        start_angle = struct.unpack('<H', packet[4:6])[0] / 100.0  # Convert to degrees
        end_angle = struct.unpack('<H', packet[22:24])[0] / 100.0  # Convert to degrees
        timestamp = struct.unpack('<H', packet[24:26])[0]
        
        # Perform CRC check
        # if not self.crc_check(packet):
        #     print("CRC check failed")
        #     return None
        
        # Parse point cloud data
        points = []
        for i in range(5):  # 12 points per packet
            # Each point is 3 bytes
            point_start = 6 + i * 3
            point_data = packet[point_start:point_start + 3]
            
            # Extract distance and angle
            distance = ((point_data[1] & 0x7F) << 8 | point_data[0]) / 1.0  # mm
            confidence = point_data[2]
            
            # Calculate angle for this point
            angle_step = (end_angle - start_angle) / 11 if end_angle != start_angle else 0
            point_angle = start_angle + i * angle_step
            
            # Convert polar to Cartesian coordinates
            x = distance * np.cos(np.radians(point_angle))
            y = distance * np.sin(np.radians(point_angle))
            
            points.append({
                'distance': distance,
                'angle': point_angle,
                'x': x,
                'y': y,
                'confidence': confidence
            })
            
            # Store for visualization
            self.x_data.append(x)
            self.y_data.append(y)
            self.confidence_data.append(confidence)
        
        return {
            'speed': speed,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'timestamp': timestamp,
            'points': points
        }
    
    def update_plot(self, frame):
        """
        Update the point cloud plot
        
        :param frame: Animation frame (unused)
        :return: Updated scatter plot
        """
        # Clear previous data
        self.scatter.set_offsets(np.c_[self.x_data, self.y_data])
        self.scatter.set_array(np.array(self.confidence_data))
        
        return self.scatter,
    
    def listen(self):
        """
        Listen to LiDAR data and parse packets with real-time visualization
        """
        try:
            # Open the serial port
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"Listening to LiDAR data on {self.serial_port}")
            
            # Setup animation for real-time plotting
            anim = FuncAnimation(self.fig, self.update_plot, 
                                 interval=50,  # Update every 50ms
                                 blit=True)
            
            # Buffer to accumulate incoming bytes
            packet_buffer = bytearray()
            
            for i in range(2000):
                # Read available bytes
                if self.ser.in_waiting > 0:
                    incoming_data = self.ser.read(self.ser.in_waiting)
                    packet_buffer.extend(incoming_data)
                    
                    # Look for complete packets
                    while len(packet_buffer) >= 28:  # Minimum full packet size
                        # Find packet start (header 0x54)
                        header_index = packet_buffer.find(b'\x54')
                        
                        if header_index == -1:
                            # No valid header found, clear buffer
                            packet_buffer.clear()
                            break
                        
                        # Remove bytes before header
                        if header_index > 0:
                            packet_buffer = packet_buffer[header_index:]
                        
                        # Check if we have a full packet
                        if len(packet_buffer) < 28:
                            break
                        
                        # Extract full packet
                        full_packet = bytes(packet_buffer[:28])
                        
                        # Parse the packet
                        parsed_data = self.parse_packet(full_packet)
                        
                        # Remove processed packet from buffer
                        packet_buffer = packet_buffer[28:]
                
                # Ensure matplotlib doesn't block
                plt.pause(0.001)
        
        except serial.SerialException as e:
            print(f"Error opening or reading from {self.serial_port}: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial port closed.")
            plt.close()

# Usage
if __name__ == "__main__":
    lidar_parser = LiDARDataParser()
    lidar_parser.listen()