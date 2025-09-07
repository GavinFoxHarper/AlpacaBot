"""
Virtual Mouse Controller - Control mouse with keyboard
Arrow keys or Numpad for movement
"""

import sys
import time
import threading
from pynput import mouse, keyboard
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, KeyCode, Listener as KeyboardListener

class VirtualMouse:
    def __init__(self):
        self.mouse = MouseController()
        self.running = True
        self.speed = 10  # Base movement speed
        self.fast_speed = 50  # Speed when holding shift
        self.slow_speed = 2  # Speed when holding ctrl
        
        # Movement states
        self.keys_pressed = set()
        self.shift_pressed = False
        self.ctrl_pressed = False
        
        print("=" * 50)
        print("       VIRTUAL MOUSE CONTROLLER")
        print("=" * 50)
        print("\nCONTROLS:")
        print("-" * 30)
        print("MOVEMENT:")
        print("  Arrow Keys or Numpad = Move mouse")
        print("  8/Up    = Move up")
        print("  2/Down  = Move down")
        print("  4/Left  = Move left")
        print("  6/Right = Move right")
        print("  7 = Move up-left")
        print("  9 = Move up-right")
        print("  1 = Move down-left")
        print("  3 = Move down-right")
        print("\nSPEED:")
        print("  Hold SHIFT = Fast movement")
        print("  Hold CTRL  = Slow/precise movement")
        print("\nCLICKS:")
        print("  5/Space    = Left click")
        print("  0/Enter    = Right click")
        print("  + (Plus)   = Double click")
        print("  Page Up    = Scroll up")
        print("  Page Down  = Scroll down")
        print("\nOTHER:")
        print("  [ = Decrease speed")
        print("  ] = Increase speed")
        print("  ESC = Exit program")
        print("-" * 30)
        print(f"\nCurrent speed: {self.speed}")
        print("Mouse controller active...\n")
        
    def on_press(self, key):
        """Handle key press events"""
        try:
            # Check for modifier keys
            if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
                self.shift_pressed = True
            elif key == Key.ctrl or key == Key.ctrl_l or key == Key.ctrl_r:
                self.ctrl_pressed = True
            
            # Add to pressed keys set
            self.keys_pressed.add(key)
            
            # Handle clicks and special actions
            if key == Key.space or (hasattr(key, 'vk') and key.vk == 101):  # Space or Numpad 5
                self.mouse.click(Button.left)
                print("Left click")
            elif key == Key.enter or (hasattr(key, 'vk') and key.vk == 96):  # Enter or Numpad 0
                self.mouse.click(Button.right)
                print("Right click")
            elif hasattr(key, 'char'):
                if key.char == '+':
                    self.mouse.click(Button.left, 2)
                    print("Double click")
                elif key.char == '[':
                    self.speed = max(1, self.speed - 5)
                    print(f"Speed decreased to: {self.speed}")
                elif key.char == ']':
                    self.speed = min(100, self.speed + 5)
                    print(f"Speed increased to: {self.speed}")
            elif key == Key.page_up:
                self.mouse.scroll(0, 3)
                print("Scroll up")
            elif key == Key.page_down:
                self.mouse.scroll(0, -3)
                print("Scroll down")
            elif key == Key.esc:
                print("\nExiting Virtual Mouse...")
                self.running = False
                return False
                
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events"""
        try:
            # Check for modifier keys
            if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
                self.shift_pressed = False
            elif key == Key.ctrl or key == Key.ctrl_l or key == Key.ctrl_r:
                self.ctrl_pressed = False
            
            # Remove from pressed keys set
            self.keys_pressed.discard(key)
        except:
            pass
    
    def move_mouse(self):
        """Continuous mouse movement based on pressed keys"""
        while self.running:
            dx, dy = 0, 0
            
            # Determine current speed
            if self.ctrl_pressed:
                current_speed = self.slow_speed
            elif self.shift_pressed:
                current_speed = self.fast_speed
            else:
                current_speed = self.speed
            
            # Check each possible movement key
            for key in self.keys_pressed:
                # Arrow keys
                if key == Key.up:
                    dy -= current_speed
                elif key == Key.down:
                    dy += current_speed
                elif key == Key.left:
                    dx -= current_speed
                elif key == Key.right:
                    dx += current_speed
                
                # Numpad keys (using vk codes)
                elif hasattr(key, 'vk'):
                    if key.vk == 104:  # Numpad 8
                        dy -= current_speed
                    elif key.vk == 98:  # Numpad 2
                        dy += current_speed
                    elif key.vk == 100:  # Numpad 4
                        dx -= current_speed
                    elif key.vk == 102:  # Numpad 6
                        dx += current_speed
                    elif key.vk == 103:  # Numpad 7
                        dx -= current_speed
                        dy -= current_speed
                    elif key.vk == 105:  # Numpad 9
                        dx += current_speed
                        dy -= current_speed
                    elif key.vk == 97:  # Numpad 1
                        dx -= current_speed
                        dy += current_speed
                    elif key.vk == 99:  # Numpad 3
                        dx += current_speed
                        dy += current_speed
            
            # Move the mouse if there's movement
            if dx != 0 or dy != 0:
                current_pos = self.mouse.position
                new_x = current_pos[0] + dx
                new_y = current_pos[1] + dy
                self.mouse.position = (new_x, new_y)
            
            # Small delay to control update rate
            time.sleep(0.01)
    
    def run(self):
        """Start the virtual mouse controller"""
        # Start movement thread
        movement_thread = threading.Thread(target=self.move_mouse, daemon=True)
        movement_thread.start()
        
        # Start keyboard listener
        with KeyboardListener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()

def main():
    try:
        # Check if required package is installed
        import pynput
    except ImportError:
        print("Installing required package: pynput...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
        print("Package installed. Please restart the program.")
        input("Press Enter to exit...")
        return
    
    # Run the virtual mouse
    vm = VirtualMouse()
    vm.run()

if __name__ == "__main__":
    main()