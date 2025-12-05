import cv2
import subprocess
import threading
import queue
import time
import numpy as np

def get_youtube_stream_url(youtube_url):
    """Use yt-dlp to get the direct stream URL"""
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]/best', '-g', youtube_url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting stream URL: {e.stderr}")
        return None

def frame_reader(ffmpeg_process, frame_queue, width, height, stop_event):
    """Reads raw video frames from ffmpeg's stdout and puts them in a queue."""
    frame_size = width * height * 3
    while not stop_event.is_set():
        # Read a single frame's worth of bytes
        in_bytes = ffmpeg_process.stdout.read(frame_size)
        if len(in_bytes) == 0:
            # End of stream
            break
        
        if len(in_bytes) != frame_size:
            print(f"Warning: Incomplete frame received. Expected {frame_size}, got {len(in_bytes)}")
            continue

        # Convert the bytes to a numpy array and reshape
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Drop oldest frame
            except queue.Empty:
                pass
        
        frame_queue.put(frame)
    
    frame_queue.put(None) # Signal that the reader has finished

# --- Main ---
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=RH5fgOcO0jg"
    
    # Define the resolution you want to process
    WIDTH, HEIGHT = 1280, 720

    print("Getting YouTube stream URL...")
    stream_url = get_youtube_stream_url(youtube_url)

    if not stream_url:
        print("Could not get stream URL. Exiting.")
        exit()

    # FFmpeg command to read the stream and pipe raw video to stdout
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',                 # Read at native frame rate
        '-i', stream_url,
        '-fflags', 'nobuffer', # Reduce latency
        '-probesize', '32',    # Analyze stream quickly
        '-f', 'rawvideo',      # Output format
        '-pix_fmt', 'bgr24',   # Pixel format OpenCV likes
        '-vf', f'scale={WIDTH}:{HEIGHT}', # Scale the video
        '-loglevel', 'error',  # Suppress verbose output
        '-'                    # Output to stdout
    ]

    print("Starting ffmpeg process...")
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

    # A queue to hold frames from the reader thread
    frame_queue = queue.Queue(maxsize=60) # Increased buffer size
    
    # An event to signal the reader thread to stop
    stop_event = threading.Event()

    # Start the frame reader thread
    reader_thread = threading.Thread(
        target=frame_reader, 
        args=(ffmpeg_process, frame_queue, WIDTH, HEIGHT, stop_event)
    )
    reader_thread.daemon = True
    reader_thread.start()

    print("Playback started. Press 'q' to quit.")

    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            
            cv2.imshow('Frame', frame)

        except queue.Empty:
            if not reader_thread.is_alive():
                break
            print("Waiting for frames...")
            time.sleep(0.1)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Cleaning up...")
    stop_event.set()
    # Terminate ffmpeg if it's still running
    if ffmpeg_process.poll() is None:
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
    
    reader_thread.join(timeout=2)
    cv2.destroyAllWindows()