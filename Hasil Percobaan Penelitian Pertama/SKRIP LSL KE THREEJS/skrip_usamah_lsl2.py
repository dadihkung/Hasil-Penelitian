import time
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Keys to test sending, mapping strings to pynput keys where needed
test_keys = ['w', 'a', 's', 'd', 'q', 'Shift', ' ']

def convert_key(k):
    special_keys = {
        'Shift': Key.shift,
        'Space': Key.space,
        'Enter': Key.enter,
        # add more if needed
    }
    return special_keys.get(k, k)

print("Starting long key press test for 3 minutes. Please focus the drone simulator window...")

start_time = time.time()
duration = 3 * 60  # 3 minutes in seconds
hold_time = 2      # how long to hold each key in seconds
pause_time = 0.5   # pause between key releases and next key press

try:
    while time.time() - start_time < duration:
        for key in test_keys:
            if time.time() - start_time >= duration:
                break
            key_to_press = convert_key(key)
            print(f"Holding key: {key} for {hold_time} seconds")
            keyboard.press(key_to_press)
            time.sleep(hold_time)
            keyboard.release(key_to_press)
            print(f"Released key: {key}")
            time.sleep(pause_time)

except KeyboardInterrupt:
    print("Keyboard test stopped.")

print("Finished long key press test.")
