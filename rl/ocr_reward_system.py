#!/usr/bin/env python
"""
OCR-based Reward System for ACT RL Training

This script continuously monitors a specified screen area using OCR to detect score changes
and communicates the scores to Unity client via UDP for reward calculation.
"""

import cv2
import numpy as np
import mss
import pytesseract
import socket
import time
import threading
import json
import argparse
import sys
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import re


class OCRRewardSystem:
    """
    OCR-based reward system that monitors screen area and sends scores via UDP.
    """
    
    def __init__(self, 
                 tesseract_path=r'D:\Tesseract-OCR\tesseract.exe',
                 unity_host='127.0.0.1',
                 unity_port=12345,
                 ocr_frequency=10,
                 screen_region=None):
        """
        Initialize the OCR reward system.
        
        Args:
            tesseract_path: Path to Tesseract OCR executable
            unity_host: Unity client host address
            unity_port: Unity client UDP port
            ocr_frequency: OCR processing frequency in Hz
            screen_region: Dictionary with 'top', 'left', 'width', 'height' keys
        """
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Network configuration
        self.unity_host = unity_host
        self.unity_port = unity_port
        
        # OCR configuration
        self.ocr_frequency = ocr_frequency
        self.ocr_interval = 1.0 / ocr_frequency  # Convert Hz to seconds
        
        # Screen region configuration
        self.screen_region = screen_region
        
        # UDP socket for communication with Unity
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # State tracking
        self.running = False
        self.current_score = 0
        self.last_valid_score = 0
        self.last_score_time = time.time()
        self.score_history = []
        
        # Threading
        self.ocr_thread = None
        self.gui_thread = None
        
        print(f"OCR Reward System initialized:")
        print(f"  - Tesseract: {tesseract_path}")
        print(f"  - Unity: {unity_host}:{unity_port}")
        print(f"  - OCR Frequency: {ocr_frequency} Hz")
        print(f"  - Screen Region: {screen_region}")
        
    def select_screen_region(self):
        """
        Interactive screen region selection using mouse.
        """
        from tkinter import Tk, Canvas
        
        region = {}
        
        def on_mouse_down(event):
            region['start_x'] = event.x
            region['start_y'] = event.y
            
        def on_mouse_up(event):
            end_x, end_y = event.x, event.y
            region['top'] = min(region['start_y'], end_y)
            region['left'] = min(region['start_x'], end_x)
            region['width'] = abs(end_x - region['start_x'])
            region['height'] = abs(end_y - region['start_y'])
            root.destroy()
            
        root = Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-alpha", 0.3)
        root.attributes("-topmost", True)
        canvas = Canvas(root, cursor="cross", bg='red')
        canvas.pack(fill="both", expand=True)
        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        
        # Add instruction text
        canvas.create_text(
            root.winfo_screenwidth() // 2, 50,
            text="Click and drag to select the score area, then release",
            fill="white", font=("Arial", 16)
        )
        
        root.mainloop()
        
        if 'width' in region and 'height' in region:
            self.screen_region = region
            print(f"Selected region: {region}")
            return region
        else:
            print("No region selected")
            return None
            
    def preprocess_image_for_ocr(self, image):
        """
        Enhance image quality for better OCR accuracy.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(thresh, -1, kernel)
        
        # Resize image for better OCR (scale up small text)
        height, width = sharpened.shape
        if height < 50 or width < 100:
            scale_factor = max(2, 100 // min(height, width))
            sharpened = cv2.resize(sharpened, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        return sharpened
        
    def extract_score_from_text(self, text):
        """
        Extract integer score from OCR text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            int: Extracted score or 0 if no valid score found
        """
        if not text or not text.strip():
            return 0
            
        # Clean the text
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        
        # Try to find integers in the text
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            # Take the largest number (assuming it's the score)
            scores = [int(num) for num in numbers]
            return max(scores)
        
        return 0
        
    def capture_and_process_score(self):
        """
        Capture screen region and extract score using OCR.
        
        Returns:
            int: Detected score or 0 if detection failed
        """
        if not self.screen_region:
            return 0
            
        try:
            with mss.mss() as sct:
                # Capture the screen region
                screenshot = sct.grab(self.screen_region)
                
                # Convert to numpy array
                frame = np.array(screenshot)
                
                # Convert to BGR format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Preprocess for OCR
                processed_frame = self.preprocess_image_for_ocr(frame)
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(
                    processed_frame, 
                    config='--psm 8 -c tessedit_char_whitelist=0123456789'
                ).strip()
                
                # Extract score
                score = self.extract_score_from_text(ocr_text)
                
                # Debug output
                if ocr_text:
                    print(f"OCR: '{ocr_text}' -> Score: {score}")
                
                return score
                
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return 0
            
    def send_score_to_unity(self, score):
        """
        Send score to Unity client via UDP.
        
        Args:
            score: Integer score to send
        """
        try:
            message = {
                'type': 'score_update',
                'score': score,
                'timestamp': time.time()
            }
            
            json_data = json.dumps(message)
            self.udp_socket.sendto(json_data.encode('utf-8'), (self.unity_host, self.unity_port))
            
        except Exception as e:
            print(f"Error sending score to Unity: {e}")
            
    def ocr_loop(self):
        """
        Main OCR processing loop running at specified frequency.
        """
        print(f"Starting OCR loop at {self.ocr_frequency} Hz...")
        
        while self.running:
            start_time = time.time()
            
            # Capture and process score
            score = self.capture_and_process_score()
            
            # Update state
            if score > 0:
                self.current_score = score
                self.last_valid_score = score
                self.last_score_time = time.time()
            else:
                self.current_score = 0
                
            # Send score to Unity
            self.send_score_to_unity(self.current_score)
            
            # Store in history
            self.score_history.append({
                'score': self.current_score,
                'timestamp': time.time()
            })
            
            # Limit history size
            if len(self.score_history) > 1000:
                self.score_history = self.score_history[-500:]
                
            # Maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.ocr_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        print("OCR loop stopped")
        
    def start(self):
        """
        Start the OCR reward system.
        """
        if not self.screen_region:
            print("No screen region configured. Please select a region first.")
            return False
            
        self.running = True
        
        # Start OCR thread
        self.ocr_thread = threading.Thread(target=self.ocr_loop, daemon=True)
        self.ocr_thread.start()
        
        print("OCR Reward System started")
        return True
        
    def stop(self):
        """
        Stop the OCR reward system.
        """
        self.running = False
        
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=2.0)
            
        # Close UDP socket
        try:
            self.udp_socket.close()
        except:
            pass
            
        print("OCR Reward System stopped")
        
    def get_statistics(self):
        """
        Get current statistics.
        """
        return {
            'current_score': self.current_score,
            'last_valid_score': self.last_valid_score,
            'last_score_time': self.last_score_time,
            'history_length': len(self.score_history),
            'running': self.running
        }


def create_control_gui(ocr_system):
    """
    Create a simple GUI for controlling the OCR system.
    """
    root = tk.Tk()
    root.title("OCR Reward System Control")
    root.geometry("400x300")
    
    # Status variables
    status_var = tk.StringVar(value="Stopped")
    score_var = tk.StringVar(value="Score: 0")
    
    def update_status():
        """Update status display"""
        if ocr_system.running:
            status_var.set("Running")
            stats = ocr_system.get_statistics()
            score_var.set(f"Score: {stats['current_score']}")
        else:
            status_var.set("Stopped")
            score_var.set("Score: 0")
        root.after(1000, update_status)  # Update every second
    
    def select_region():
        """Select screen region"""
        root.withdraw()  # Hide main window
        region = ocr_system.select_screen_region()
        root.deiconify()  # Show main window
        if region:
            messagebox.showinfo("Region Selected", f"Region: {region}")
    
    def start_ocr():
        """Start OCR system"""
        if ocr_system.start():
            messagebox.showinfo("Started", "OCR system started successfully")
        else:
            messagebox.showerror("Error", "Failed to start OCR system")
    
    def stop_ocr():
        """Stop OCR system"""
        ocr_system.stop()
        messagebox.showinfo("Stopped", "OCR system stopped")
    
    # GUI elements
    tk.Label(root, text="OCR Reward System Control", font=("Arial", 16)).pack(pady=10)
    
    tk.Label(root, textvariable=status_var, font=("Arial", 12)).pack(pady=5)
    tk.Label(root, textvariable=score_var, font=("Arial", 12)).pack(pady=5)
    
    tk.Button(root, text="Select Screen Region", command=select_region, width=20).pack(pady=5)
    tk.Button(root, text="Start OCR", command=start_ocr, width=20).pack(pady=5)
    tk.Button(root, text="Stop OCR", command=stop_ocr, width=20).pack(pady=5)
    
    # Configuration display
    config_frame = tk.Frame(root)
    config_frame.pack(pady=10)
    
    tk.Label(config_frame, text=f"Unity: {ocr_system.unity_host}:{ocr_system.unity_port}").pack()
    tk.Label(config_frame, text=f"Frequency: {ocr_system.ocr_frequency} Hz").pack()
    
    # Start status updates
    update_status()
    
    return root


def main():
    parser = argparse.ArgumentParser(description='OCR Reward System for ACT RL Training')
    parser.add_argument('--tesseract-path', type=str, default=r'D:\Tesseract-OCR\tesseract.exe',
                       help='Path to Tesseract OCR executable')
    parser.add_argument('--unity-host', type=str, default='127.0.0.1',
                       help='Unity client host address')
    parser.add_argument('--unity-port', type=int, default=12345,
                       help='Unity client UDP port')
    parser.add_argument('--ocr-frequency', type=int, default=10,
                       help='OCR processing frequency in Hz')
    parser.add_argument('--region-top', type=int, help='Screen region top coordinate')
    parser.add_argument('--region-left', type=int, help='Screen region left coordinate')
    parser.add_argument('--region-width', type=int, help='Screen region width')
    parser.add_argument('--region-height', type=int, help='Screen region height')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    # Configure screen region
    screen_region = None
    if all([args.region_top is not None, args.region_left is not None, 
            args.region_width is not None, args.region_height is not None]):
        screen_region = {
            'top': args.region_top,
            'left': args.region_left,
            'width': args.region_width,
            'height': args.region_height
        }
    
    # Create OCR system
    ocr_system = OCRRewardSystem(
        tesseract_path=args.tesseract_path,
        unity_host=args.unity_host,
        unity_port=args.unity_port,
        ocr_frequency=args.ocr_frequency,
        screen_region=screen_region
    )
    
    if args.no_gui:
        # Command line mode
        print("Running in command line mode...")
        
        if not screen_region:
            print("No screen region specified. Use --region-* arguments or run with GUI.")
            return
            
        try:
            ocr_system.start()
            print("Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            ocr_system.stop()
    else:
        # GUI mode
        try:
            root = create_control_gui(ocr_system)
            root.protocol("WM_DELETE_WINDOW", lambda: (ocr_system.stop(), root.destroy()))
            root.mainloop()
        except Exception as e:
            print(f"GUI error: {e}")
            ocr_system.stop()


if __name__ == "__main__":
    main()
