# 
# 
# SeedPointToMaskOut
import tkinter as tk
from tkinter import filedialog, Canvas, Scale, Button, Entry, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
from pdf2image import convert_from_path
import re
import pytesseract
import tifffile
import math


class FloodFillApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flood Fill PDF Mask Generator")

        self.canvas_width = 800
        self.canvas_height = 600

        # Main vertical layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Image preview pane
        self.canvas = Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom: Horizontal layout for buttons and table
        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Left: Button panel
        self.button_frame = tk.Frame(self.bottom_frame)
        self.button_frame.pack(side=tk.LEFT, anchor=tk.SW, padx=10, pady=10)

        self.load_button = Button(self.button_frame, text="Load PDF", command=self.load_pdf)
        self.load_button.pack()

        # Aggressiveness slider
        self.aggressiveness_slider = Scale(self.button_frame, from_=60, to=180, orient=tk.HORIZONTAL, label="Aggressiveness")
        self.aggressiveness_slider.set(70)
        self.aggressiveness_slider.pack()

        self.clear_button = Button(self.button_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()
        
        self.fit_button = Button(self.button_frame, text="Fit to View", command=self.fit_to_view)
        self.fit_button.pack()

        #self.contour_button = Button(self.button_frame, text="Show Outer Contour", command=self.find_outer_contour)
        #self.contour_button.pack()
        
        self.pixel_slider = Scale(self.button_frame, from_=1, to=30, orient=tk.HORIZONTAL, label="Gap Pixels")
        self.pixel_slider.set(15)
        self.pixel_slider.pack()

        #self.fill_gaps_button = Button(self.button_frame, text="Fill Gaps in Contour", command=self.recompute_contour_with_closing)
        #self.fill_gaps_button.pack()

        self.extract_button = Button(self.button_frame, text="Extract Distances", command=self.extract_text_along_decimated_lines)
        self.extract_button.pack()


        self.scale_input_frame = tk.Frame(self.bottom_frame)
        self.scale_input_frame.pack(side=tk.BOTTOM, anchor=tk.SW, padx=10, pady=5)

        tk.Label(self.scale_input_frame, text="Real-world Distance (feet):", font=("Arial", 10)).pack(side=tk.LEFT)
        self.real_distance_entry = Entry(self.scale_input_frame, width=10)
        self.real_distance_entry.pack(side=tk.LEFT, padx=5)

        Button(self.scale_input_frame, text="Calc Scale Factor", command=lambda: self.calculate_scale_factor()).pack(side=tk.LEFT)

        self.scale_factor_label = Label(self.bottom_frame, text="Scale Factor: Not set", font=("Arial", 10), fg="blue")
        self.scale_factor_label.pack(side=tk.BOTTOM, anchor=tk.SW, padx=10, pady=5)

        self.line_points = []
        self.SCALE_FACTOR = None
        self.canvas.bind("<Button-3>", self.trace_line)  # Right-click to trace

        self.export_geotiff_button = Button(self.button_frame, text="Export GeoTIFF", command=self.export_geotiff)
        self.export_geotiff_button.pack()

        # Add slider for contour decimation (simplification)
        self.simplify_slider = Scale(
            self.button_frame,
            from_=0.001, to=0.01,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            label="Simplify Contour"
        )
        self.simplify_slider.set(0.001)
        self.simplify_slider.pack(side=tk.LEFT)

        # Add button to create simplified contour
        self.simplify_button = Button(
            self.button_frame,
            text="Create Simplified Contour",
            command=self.on_simplify_contour
        )
        self.simplify_button.pack(side=tk.LEFT)

        # Add export button for decimated mask image
        self.export_decmask_button = Button(
            self.button_frame,
            text="Export DecMaskImage.png",
            command=self.export_decmask_image
        )
        self.export_decmask_button.pack(side=tk.LEFT)

        # Right: Distance table
        self.table_frame = tk.Frame(self.bottom_frame)
        self.table_frame.pack(side=tk.RIGHT, anchor=tk.SE, padx=10, pady=10)

        self.table_label = tk.Label(self.table_frame, text="Distances", font=("Arial", 12, "bold"))
        self.table_label.pack()

        self.table_text = tk.Text(self.table_frame, width=80, height=12, font=("Arial", 10))
        self.table_text.pack()

        # Mouse coordinates label (move to right below table)
        self.coord_label = Label(self.table_frame, text="Mouse: (x, y)", font=("Arial", 10), fg="black")
        self.coord_label.pack(side=tk.BOTTOM, anchor=tk.SE, padx=10, pady=2)

        # Canvas interaction variables
        self.image = None
        self.tk_image = None
        self.seed_points = []
        self.mask = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_pos = None

        # Bind canvas interactions
        self.canvas.bind("<Button-1>", self.add_seed_point)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.reset_mouse_pos)

        self.canvas.bind("<Configure>", self.on_resize)

        # Bind mouse motion event to canvas
        self.canvas.bind("<Motion>", self.show_mouse_coords)

        self.ocr_candidate_boxes = []

        # Autoload input.pdf on startup
        self.autoload_pdf("input.pdf")

        self.decimated_contour = None  # Store simplified contour points
        self.decimated_epsilon = None  # Store epsilon used for decimation

        # Ensure preview updates when slider moves
        self.aggressiveness_slider.config(command=lambda v: self.update_preview())

        # Add regex pattern fields for OCR candidate extraction
        self.regex_frame = tk.Frame(self.button_frame)
        self.regex_frame.pack(pady=5)

        tk.Label(self.regex_frame, text="Direction Regex:").pack(side=tk.LEFT)
        self.direction_regex_entry = Entry(self.regex_frame, width=20)
        self.direction_regex_entry.pack(side=tk.LEFT, padx=2)
        #self.direction_regex_entry.insert(0, r"^[NS].*[WE]$")

        tk.Label(self.regex_frame, text="Distance Regex:").pack(side=tk.LEFT)
        self.distance_regex_entry = Entry(self.regex_frame, width=20)
        self.distance_regex_entry.pack(side=tk.LEFT, padx=2)
        self.distance_regex_entry.insert(0, r"\d{1,4}\.\d{1,3}\s?'")


    def autoload_pdf(self, filename):
        import os
        if os.path.exists(filename):
            try:
                dpi = 200
                pages = convert_from_path(
                    filename,
                    dpi=dpi,
                    poppler_path=r'C:\\Apps\\poppler-24.08.0\\Library\\bin'
                )
                pil_image = pages[0]
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                h, w = self.original_image.shape[:2]
                self.viewport = Viewport(w, h, self.canvas_width, self.canvas_height)
                self.image = self.original_image.copy()
                self.seed_points.clear()
                self.mask = None
                self.fit_to_view()
                print(f"Autoloaded {filename}")
            except Exception as e:
                print(f"Failed to autoload {filename}: {e}")
        else:
            print(f"File {filename} not found.")

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            dpi = 400
            pages = convert_from_path(
                file_path,
                dpi=dpi,
                poppler_path=r'C:\\Apps\\poppler-24.08.0\\Library\\bin'
            )
            pil_image = pages[0]
            self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            h, w = self.original_image.shape[:2]

            # Initialize viewport
            self.viewport = Viewport(w, h, self.canvas_width, self.canvas_height)

            self.image = self.original_image.copy()
            self.seed_points.clear()
            self.mask = None
            self.fit_to_view()

    def add_seed_point(self, event):
        if not hasattr(self, 'viewport') or self.viewport is None:
            tk.messagebox.showwarning("Image Not Loaded", "Please load an image before adding seed points.")
            return

        x_canvas, y_canvas = event.x, event.y
        x_img = int(self.viewport.offset_x + x_canvas / self.viewport.zoom)
        y_img = int(self.viewport.offset_y + y_canvas / self.viewport.zoom)

        if self.image is not None:
            h, w = self.image.shape[:2]
            if 0 <= x_img < w and 0 <= y_img < h:
                self.seed_points.append((x_img, y_img))
                self.apply_flood_fill()
                self.update_preview()
                # Automatically run contour and gap fill after seed point
                self.find_outer_contour()
                self.recompute_contour_with_closing()

    def apply_flood_fill(self):
        if self.image is None or not self.seed_points:
            return

        flood_img = self.image.copy()
        h, w = flood_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        aggressiveness = self.aggressiveness_slider.get()

        for point in self.seed_points:
            # Use FIXED_RANGE to ensure initial fill works
            cv2.floodFill(
                flood_img, mask, point, (255, 255, 255),
                (aggressiveness,) * 3, (aggressiveness,) * 3,
                flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )

        # Remove border and scale mask for display
        mask = mask[1:-1, 1:-1]
        if np.count_nonzero(mask) == 0:
            print("Flood fill did not mark any mask pixels. Try adjusting aggressiveness or seed point.")
        self.mask = mask * 255

    def update_preview(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
        if not hasattr(self, 'viewport') or self.viewport is None:
            return

        preview = self.viewport.get_view(self.original_image.copy())

        # Inverse correlation between aggressiveness and threshold
        max_aggressiveness = self.aggressiveness_slider.cget("to")
        min_threshold = 50
        max_threshold = 150
        aggressiveness = self.aggressiveness_slider.get()
        threshold_value = int(max_threshold - (aggressiveness - 1) * (max_threshold - min_threshold) / (max_aggressiveness - 1))

        if threshold_value > 0:
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            preview = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

        # Apply mask if available
        if self.mask is not None and self.mask.size > 0:
            # Calculate viewport region in image coordinates
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            view_w = int(canvas_w / self.viewport.zoom)
            view_h = int(canvas_h / self.viewport.zoom)

            x1 = int(self.viewport.offset_x)
            y1 = int(self.viewport.offset_y)
            x2 = min(x1 + view_w, self.mask.shape[1])
            y2 = min(y1 + view_h, self.mask.shape[0])

            # Crop mask to viewport region
            cropped_mask = self.mask[y1:y2, x1:x2]
            # Resize mask to canvas size
            if cropped_mask.size > 0:
                resized_mask = cv2.resize(cropped_mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
                # Overlay mask: set preview pixels to green where mask is 255
                preview[resized_mask == 255] = [0, 255, 0]

        # Draw seed points
        for x, y in self.seed_points:
            canvas_x = int((x - self.viewport.offset_x) * self.viewport.zoom)
            canvas_y = int((y - self.viewport.offset_y) * self.viewport.zoom)
            cv2.circle(preview, (canvas_x, canvas_y), radius=2, color=(0, 0, 255), thickness=-1)

        # Draw outer contour if available
        if hasattr(self, 'outer_contour') and self.outer_contour is not None:
            for cnt in self.outer_contour:
                transformed = np.array([
                    [(pt[0] - self.viewport.offset_x) * self.viewport.zoom,
                     (pt[1] - self.viewport.offset_y) * self.viewport.zoom]
                    for pt in cnt.reshape(-1, 2)
                ], dtype=np.int32)
                cv2.drawContours(preview, [transformed], -1, (0, 0, 255), 2)

        # Draw persistent OCR candidate bounding boxes in blue
        if hasattr(self, 'ocr_candidate_boxes'):
            for x_abs, y_abs, w_abs, h_abs, angle in self.ocr_candidate_boxes:
                x1 = int((x_abs - self.viewport.offset_x) * self.viewport.zoom)
                y1 = int((y_abs - self.viewport.offset_y) * self.viewport.zoom)
                x2 = int((x_abs + w_abs - self.viewport.offset_x) * self.viewport.zoom)
                y2 = int((y_abs + h_abs - self.viewport.offset_y) * self.viewport.zoom)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue

        # Draw persistent decimated contour if present
        if self.decimated_contour is not None:
            # Transform decimated contour to viewport/canvas coordinates
            transformed = np.array([
                [(pt[0] - self.viewport.offset_x) * self.viewport.zoom,
                 (pt[1] - self.viewport.offset_y) * self.viewport.zoom]
                for pt in self.decimated_contour.reshape(-1, 2)
            ], dtype=np.int32)
            for i in range(len(transformed)):
                pt1 = tuple(transformed[i])
                pt2 = tuple(transformed[(i + 1) % len(transformed)])
                cv2.line(preview, pt1, pt2, (255, 255, 0), 2)  # Cyan lines
            # Draw best fit lines (blue) in viewport
            for i in range(len(transformed)):
                segment = [transformed[i], transformed[(i + 1) % len(transformed)]
                ]
                segment_np = np.array(segment, dtype=np.int32)
                [vx, vy, x0, y0] = cv2.fitLine(segment_np, cv2.DIST_L2, 0, 0.01, 0.01)
                length = 1000
                x1 = int(x0 - vx * length)
                y1 = int(y0 - vy * length)
                x2 = int(x0 + vx * length)
                y2 = int(y0 + vy * length)
                cv2.line(preview, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Convert to Tkinter image
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_mask(self):
        if self.mask is None:
            print("No mask available.")
            return

        # Apply morphological closing to fill small gaps
        pixel_count = self.pixel_slider.get() if hasattr(self, 'pixel_slider') else 5
        kernel = np.ones((pixel_count, pixel_count), np.uint8)
        closed_mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # Find outer contours from the closed mask
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found.")
            return

        # Create a new mask with filled contour
        h, w = closed_mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Convert to RGB for saving as JPEG
        white_rgb = cv2.merge([contour_mask, contour_mask, contour_mask])
        img_pil = Image.fromarray(white_rgb)

        # Save to file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg")]
        )
        if file_path:
            img_pil.save(file_path)
            print(f"Mask saved to {file_path}")

    def clear_canvas(self):
        self.seed_points.clear()
        self.mask = None            # Clear the mask
        self.outer_contour = None   # Clear the outer contour
        self.update_preview()
        
    def zoom(self, event):
        MAX_ZOOM_LEVEL = 1.0  # Maximum zoom level for 1:1 pixel ratio
        if self.original_image is None or self.viewport is None:
            return

        # Determine zoom direction
        if hasattr(event, 'delta'):
            # Windows scroll direction
            factor = 1.1 if event.delta > 0 else 0.9  # Scroll forward = zoom in
            # Linux scroll direction
        elif event.num == 4:  # Linux scroll up
            factor = 1.1  # Zoom in
        elif event.num == 5:  # Linux scroll down
            factor = 0.9  # Zoom out
        else:
            return

        # Get mouse position on canvas
        mouse_x, mouse_y = event.x, event.y

        # Convert canvas coordinates to image coordinates
        img_x = self.viewport.offset_x + mouse_x / self.viewport.zoom
        img_y = self.viewport.offset_y + mouse_y / self.viewport.zoom

        fit_scale = min(self.canvas_width / self.original_image.shape[1],
                        self.canvas_height / self.original_image.shape[0])
        new_zoom = self.viewport.zoom * factor
        new_zoom = max(fit_scale, min(new_zoom, MAX_ZOOM_LEVEL))

        # Only update if zoom changed
        if new_zoom != self.viewport.zoom:
            self.viewport.offset_x = img_x - mouse_x / new_zoom
            self.viewport.offset_y = img_y - mouse_y / new_zoom
            self.viewport.zoom = new_zoom
            self.update_preview()

    def pan(self, event):
        if self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            dy = event.y - self.last_mouse_pos[1]

            # Move viewport in same direction as mouse
            self.viewport.offset_x -= dx / self.viewport.zoom
            self.viewport.offset_y -= dy / self.viewport.zoom

            # Ensure offsets stay within bounds
            self.viewport.offset_x = max(0, min(
                self.viewport.offset_x,
                self.viewport.image_width - self.canvas.winfo_width() / self.viewport.zoom
            ))
            self.viewport.offset_y = max(0, min(
                self.viewport.offset_y,
                self.viewport.image_height - self.canvas.winfo_height() / self.viewport.zoom
            ))

            self.update_preview()

        self.last_mouse_pos = (event.x, event.y)
  
    def reset_mouse_pos(self, event):
        self.last_mouse_pos = None
        
    def start_pan(self, event):
        self.last_mouse_pos = (event.x, event.y)

    def find_outer_contour(self):
        if self.mask is None or self.original_image is None or self.viewport is None:
            print("Mask or image not available.")
            return

        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        self.outer_contour = contours  # Store all contours
        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

        preview = self.viewport.get_view(contour_img)

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def recompute_contour_with_closing(self):
        if self.mask is None:
            print("No mask available.")
            return

        # Get pixel count from slider
        pixel_count = self.pixel_slider.get()
        kernel = np.ones((pixel_count, pixel_count), np.uint8)

        # Apply morphological closing to fill small gaps
        closed_mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # Find outer contours from the closed mask
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found.")
            return

        self.outer_contour = contours

        # Create a new mask using the filled contour
        h, w = closed_mask.shape
        new_mask = np.zeros((h + 2, w + 2), np.uint8)
        for point in self.seed_points:
            cv2.floodFill(self.original_image.copy(), new_mask, point, (255, 255, 255),
                          (self.aggressiveness_slider.get(),) * 3, (self.aggressiveness_slider.get(),) * 3,
                          flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

        self.mask = new_mask[1:-1, 1:-1] * 255
        self.update_preview()

    def preprocess_for_ocr(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding to enhance contrast
        # thresh = cv2.adaptiveThreshold(
        #     blurred, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11, 2
        # )
        thresh = blurred

        # Optional: Upscale image to improve OCR accuracy
        upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return upscaled

    def extract_distances_near_contour(self):
        if self.image is None or self.mask is None or not hasattr(self, 'outer_contour') or not self.outer_contour:
            return

        max_distance_to_contour = 200
        outer_contour = max(self.outer_contour, key=cv2.contourArea)

        # Crop image to bounding box around contour
        x, y, w, h = cv2.boundingRect(outer_contour)
        pad = max_distance_to_contour
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, self.image.shape[1])
        y2 = min(y + h + pad, self.image.shape[0])
        cropped_img = self.image[y1:y2, x1:x2]

        # Preprocess only the cropped region
        preprocessed_img = FloodFillApp.preprocess_for_ocr(cropped_img)

        # Prepare debug image: convert to BGR if needed
        if len(preprocessed_img.shape) == 2:
            debug_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = preprocessed_img.copy()

        # Transform contour coordinates to preprocessed image space
        # Account for cropping and scaling
        scale_x = cropped_img.shape[1] / preprocessed_img.shape[1]
        scale_y = cropped_img.shape[0] / preprocessed_img.shape[0]
        contour_shifted = outer_contour - [x1, y1]
        contour_scaled = np.array(contour_shifted, dtype=np.float32)
        contour_scaled[:, 0, 0] = contour_scaled[:, 0, 0] / scale_x
        contour_scaled[:, 0, 1] = contour_scaled[:, 0, 1] / scale_y
        contour_scaled = contour_scaled.astype(np.int32)

        # Draw contour as thin green line
        cv2.drawContours(debug_img, [contour_scaled], -1, (0, 255, 0), 1)

        # Save debug image
        cv2.imwrite("debug_preprocessed.png", debug_img)

        # OCR only on cropped region
        ocr_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT,
                                             config="--psm 7 -c tessedit_char_whitelist=0123456789'\"ftin ")

        scale_x = cropped_img.shape[1] / preprocessed_img.shape[1]
        scale_y = cropped_img.shape[0] / preprocessed_img.shape[0]

        highlight_img = self.image.copy()
        inside_distances = []
        outside_distances = []

        self.ocr_candidate_boxes = []  # Reset before each extraction

        # why is the only regex search happenign here?
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            # Search for numbers ending in apostrophe
            match = re.search(r"(\d+)'$", text)
            if not match:
                continue
            extracted = match.group(0)  # e.g., "12'"

            x_rel = ocr_data['left'][i]
            y_rel = ocr_data['top'][i]
            w_rel = ocr_data['width'][i]
            h_rel = ocr_data['height'][i]
            x_abs = x1 + int(x_rel * scale_x)
            y_abs = y1 + int(y_rel * scale_y)
            w_abs = int(w_rel * scale_x)
            h_abs = int(h_rel * scale_y)
            center = (x_abs + w_abs // 2, y_abs + h_abs // 2)

            # Check if center is inside or outside the mask
            mask_h, mask_w = self.mask.shape
            cx, cy = int(center[0]), int(center[1])
            is_inside_mask = (0 <= cy < mask_h) and (0 <= cx < mask_w) and (self.mask[cy, cx] > 0)

            dist = cv2.pointPolygonTest(outer_contour, center, True)
            if abs(dist) <= max_distance_to_contour:
                cv2.rectangle(highlight_img, (x_abs, y_abs), (x_abs + w_abs, y_abs + h_abs), (0, 0, 255), 2)
                if is_inside_mask:
                    inside_distances.append((extracted, center, dist))
                else:
                    outside_distances.append((extracted, center, dist))

            # Store bounding box in original image coordinates
            self.ocr_candidate_boxes.append((x_abs, y_abs, w_abs, h_abs))

        self.inside_distances = inside_distances
        self.outside_distances = outside_distances

        # Step 5: Sort distances clockwise around contour center
        valid_distances = [(text, pt) for text, pt, _ in inside_distances + outside_distances if text.strip()]
        sorted_distances = self.sort_clockwise(valid_distances)

        # Step 6: Display distances in the table (original image coordinates)
        self.table_text.delete("1.0", tk.END)
        if sorted_distances:
            for i, (text, pt) in enumerate(sorted_distances):
                dist = next((d for t, p, d in inside_distances + outside_distances if t == text and p == pt), None)
                location = "inside" if (text, pt, dist) in inside_distances else "outside"
                self.table_text.insert(tk.END, f"{i+1}. {text} at {pt}, distance to contour: {dist:.2f} ({location})\n")
        else:
            self.table_text.insert(tk.END, "No distances found near contour.")

        # Show highlights in preview
        preview = self.viewport.get_view(highlight_img)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def sort_clockwise(self, points):
        if not points:
            return []

        # Compute center of all points
        cx = np.mean([pt[0] for _, pt in points])
        cy = np.mean([pt[1] for _, pt in points])

        def angle(p):
            return np.arctan2(p[1] - cy, p[0] - cx)

        return sorted(points, key=lambda x: angle(x[1]))
 
    def fit_to_view(self):
        if self.original_image is None or self.viewport is None:
            return

        h, w = self.original_image.shape[:2]
        scale_x = self.canvas_width / w
        scale_y = self.canvas_height / h
        fit_scale = min(scale_x, scale_y)

        # Set zoom level
        self.viewport.zoom = fit_scale

        # Center the image in the canvas
        self.viewport.offset_x = (w - self.canvas_width / fit_scale) / 2
        self.viewport.offset_y = (h - self.canvas_height / fit_scale) / 2

        self.update_preview()
        
    def reset_zoom(self):
        if self.viewport is None:
            return

        # Reset zoom to 1.0 and center the image
        self.viewport.zoom = 1.0
        self.viewport.offset_x = (self.viewport.image_width - self.canvas_width) / 2
        self.viewport.offset_y = (self.viewport.image_height - self.canvas_height) / 2

        self.update_preview()
            
    def on_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

        if hasattr(self, 'viewport') and self.viewport:
            self.viewport.canvas_width = event.width
            self.viewport.canvas_height = event.height

        self.update_preview()

    def trace_line(self, event):
        x_canvas, y_canvas = event.x, event.y
        x_img = int(self.viewport.offset_x + x_canvas / self.viewport.zoom)
        y_img = int(self.viewport.offset_y + y_canvas / self.viewport.zoom)

        self.line_points.append((x_img, y_img))
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="cyan")

        if len(self.line_points) == 2:
            x1, y1 = self.line_points[0]
            x2, y2 = self.line_points[1]
            canvas_x1 = int((x1 - self.viewport.offset_x) * self.viewport.zoom)
            canvas_y1 = int((y1 - self.viewport.offset_y) * self.viewport.zoom)
            canvas_x2 = int((x2 - self.viewport.offset_x) * self.viewport.zoom)
            canvas_y2 = int((y2 - self.viewport.offset_y) * self.viewport.zoom)
            self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, fill="cyan", width=2)
            
    def calculate_scale_factor(self):
        self.line_points = []

        def on_right_click(event):
            x_canvas, y_canvas = event.x, event.y
            x_img = int(self.viewport.offset_x + x_canvas / self.viewport.zoom)
            y_img = int(self.viewport.offset_y + y_canvas / self.viewport.zoom)
            self.line_points.append((x_img, y_img))
            self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="cyan")

            if len(self.line_points) == 2:
                x1, y1 = self.line_points[0]
                x2, y2 = self.line_points[1]
                canvas_x1 = int((x1 - self.viewport.offset_x) * self.viewport.zoom)
                canvas_y1 = int((y1 - self.viewport.offset_y) * self.viewport.zoom)
                canvas_x2 = int((x2 - self.viewport.offset_x) * self.viewport.zoom)
                canvas_y2 = int((y2 - self.viewport.offset_y) * self.viewport.zoom)
                self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2, fill="cyan", width=2)

                try:
                    real_distance_feet = float(self.real_distance_entry.get())
                    print("Real-world distance is", real_distance_feet, " feet")
                except ValueError:
                    print("Invalid distance entered.")
                    return

                real_distance_meters = real_distance_feet * 0.3048
                pixel_distance = math.hypot(x2 - x1, y2 - y1)

                if real_distance_meters == 0:
                    print("Real-world distance is zero.")
                    return
                if pixel_distance == 0:
                    print("Pixel distance is zero.")
                    return

                self.SCALE_FACTOR = real_distance_meters / pixel_distance  # meters per pixel
                self.PIXEL_SCALE = pixel_distance / real_distance_meters   # pixels per meter

                self.scale_factor_label.config(
                    text=f"Scale Factor: {self.SCALE_FACTOR:.4f} meters/pixel, Pixel Scale: {self.PIXEL_SCALE:.2f} pixels/meter"
                )
                print(f"Scale factor set to {self.SCALE_FACTOR:.4f} meters/pixel")
                print(f"Pixel scale set to {self.PIXEL_SCALE:.2f} pixels/meter")

                self.canvas.unbind("<Button-3>")

        self.canvas.bind("<Button-3>", on_right_click)
        print("Right-click two points on the canvas to define the scale line.")

    def show_mouse_coords(self, event):
        # Canvas coordinates
        x_canvas, y_canvas = event.x, event.y

        # Image coordinates (if viewport exists)
        if hasattr(self, 'viewport') and self.viewport is not None:
            x_img = int(self.viewport.offset_x + x_canvas / self.viewport.zoom)
            y_img = int(self.viewport.offset_y + y_canvas / self.viewport.zoom)
            self.coord_label.config(text=f"Mouse: Canvas ({x_canvas}, {y_canvas})  Image ({x_img}, {y_img})")
        else:
            self.coord_label.config(text=f"Mouse: Canvas ({x_canvas}, {y_canvas})")

    def export_geotiff(self):
        if self.mask is None:
            print("No mask available.")
            return

        if self.SCALE_FACTOR is None:
            print("Scale factor not set.")
            return

        # Apply morphological closing to fill small gaps
        pixel_count = self.pixel_slider.get() if hasattr(self, 'pixel_slider') else 5
        kernel = np.ones((pixel_count, pixel_count), np.uint8)
        closed_mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # Find outer contours from the closed mask
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found.")
            return

        # Create a new mask with filled contour
        h, w = closed_mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Prepare GeoTIFF pixel scale tag (ModelPixelScaleTag, tag 33550)
        # Format: (tag, dtype, count, value, writeonce)
        # For a 2D image, Z scale is usually 0
        pixel_scale_tag = (33550, 'd', 3, [self.SCALE_FACTOR, self.SCALE_FACTOR, 0.0], False)

        # Save as GeoTIFF with pixel scale EXIF data
        file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif")])
        if file_path:
            tifffile.imwrite(
                file_path,
                contour_mask,
                extratags=[pixel_scale_tag]
            )
            print(f"GeoTIFF saved to {file_path} with ModelPixelScaleTag={self.SCALE_FACTOR} meters/pixel")

    def on_simplify_contour(self):
        # Only use the current outer_contour, do not recompute
        epsilon_ratio = self.simplify_slider.get()
        self.overlay_simplified_contour(epsilon_ratio)

    def overlay_simplified_contour(self, epsilon_ratio=0.01):
        if not hasattr(self, 'outer_contour') or self.outer_contour is None:
            print("No outer contour available.")
            return

        # Use the largest contour from the existing outer_contour
        contour = max(self.outer_contour, key=cv2.contourArea)
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Trim at intersection of nearest neighbor
        trimmed = self._trim_at_intersections(approx)

        self.decimated_contour = trimmed
        self.decimated_epsilon = epsilon_ratio

        print(f"Simplified contour with epsilon={epsilon:.2f} ({epsilon_ratio*100:.2f}%) from {len(contour)} to {len(trimmed)} points.")

        img = self.original_image.copy()
        self._draw_decimated_contour(img)

        preview = self.viewport.get_view(img)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def _trim_at_intersections(self, approx):
        # Remove points that are too close to their neighbors (intersection/overlap)
        if len(approx) < 3:
            return approx
        trimmed = []
        min_dist = 5  # pixels, adjust as needed
        for i in range(len(approx)):
            pt = approx[i][0]
            prev_pt = approx[i-1][0]
            next_pt = approx[(i+1) % len(approx)][0]
            dist_prev = np.linalg.norm(pt - prev_pt)
            dist_next = np.linalg.norm(pt - next_pt)
            # Keep if not too close to either neighbor
            if dist_prev > min_dist and dist_next > min_dist:
                trimmed.append([pt])
        if len(trimmed) < 3:
            # If too aggressive, fallback to original
            return approx
        return np.array(trimmed, dtype=np.int32)

    def _draw_decimated_contour(self, img):
        if self.decimated_contour is not None:
            approx = self.decimated_contour
            for i in range(len(approx)):
                pt1 = tuple(approx[i][0])
                pt2 = tuple(approx[(i + 1) % len(approx)][0])
                cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Cyan lines
            for i in range(len(approx)):
                segment = [approx[i][0], approx[(i + 1) % len(approx)][0]]
                segment_np = np.array(segment, dtype=np.int32)
                [vx, vy, x0, y0] = cv2.fitLine(segment_np, cv2.DIST_L2, 0, 0.01, 0.01)
                length = 100
                x1 = int(x0 - vx * length)
                y1 = int(y0 - vy * length)
                x2 = int(x0 + vx * length)
                y2 = int(y0 + vy * length)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
    def export_decmask_image(self):
        # Compose the onscreen overlay as seen in update_preview
        if not hasattr(self, 'original_image') or self.original_image is None:
            print("No image loaded.")
            return

        img = self.original_image.copy()

        # Overlay mask in green
        if self.mask is not None and self.mask.size > 0:
            mask_rgb = cv2.merge([self.mask, self.mask, self.mask])
            green_mask = np.zeros_like(mask_rgb)
            green_mask[:, :, 1] = mask_rgb[:, :, 1]
            img = cv2.addWeighted(img, 1.0, green_mask, 0.4, 0)

        # Overlay outer contour in red
        if hasattr(self, 'outer_contour') and self.outer_contour is not None:
            cv2.drawContours(img, self.outer_contour, -1, (0, 0, 255), 2)

        # Overlay decimated contour in cyan and best fit lines in blue
        if self.decimated_contour is not None:
            approx = self.decimated_contour
            for i in range(len(approx)):
                pt1 = tuple(approx[i][0])
                pt2 = tuple(approx[(i + 1) % len(approx)][0])
                cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Cyan lines
            for i in range(len(approx)):
                segment = [approx[i][0], approx[(i + 1) % len(approx)][0]]
                segment_np = np.array(segment, dtype=np.int32)
                [vx, vy, x0, y0] = cv2.fitLine(segment_np, cv2.DIST_L2, 0, 0.01, 0.01)
                length = 1000
                x1 = int(x0 - vx * length)
                y1 = int(y0 - vy * length)
                x2 = int(x0 + vx * length)
                y2 = int(y0 + vy * length)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue lines

        # Save as PNG
        cv2.imwrite("DecMaskImage.png", img)
        print("Exported DecMaskImage.png")

    @staticmethod
    def preprocess_roi_for_ocr(roi):
        # Convert to grayscale
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Denoise
        roi_denoised = cv2.fastNlMeansDenoising(roi_gray, None, h=30, templateWindowSize=7, searchWindowSize=21)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        roi_sharp = cv2.filter2D(roi_denoised, -1, kernel)

        # Preprocess ROI to connect characters
        kernel = np.ones((2, 2), np.uint8)
        roi_closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        # Contrast enhancement
        roi_eq = cv2.equalizeHist(roi_sharp)

        # Threshold (try both adaptive and binary)
        roi_thresh = cv2.adaptiveThreshold(
            roi_eq, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        # Optionally try: _, roi_thresh = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Upscale more aggressively
        roi_up = cv2.resize(roi_thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Pad horizontally
        pad = 20
        roi_up = cv2.copyMakeBorder(roi_up, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)

        return roi_up

    def extract_text_along_decimated_lines(self):
        if self.image is None or self.decimated_contour is None:
            return

        img = self.image.copy()
        highlight_img = self.image.copy()
        contour = self.decimated_contour
        line_width = 100
        ocr_results = []
        self.ocr_candidate_boxes = []

        # Get regex patterns from entry fields
        direction_pattern_str = self.direction_regex_entry.get()
        distance_pattern_str = self.distance_regex_entry.get()

        direction_pattern = re.compile(direction_pattern_str, re.IGNORECASE) if direction_pattern_str else None
        distance_pattern = re.compile(distance_pattern_str) if distance_pattern_str else None

        # Use the largest contour for distance calculation
        if hasattr(self, 'outer_contour') and self.outer_contour is not None:
            main_contour = max(self.outer_contour, key=cv2.contourArea)
        else:
            main_contour = contour

        max_distance_to_contour = 100  # pixels

        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]

            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = int(math.hypot(dx, dy))
            if length < 10:
                continue

            # Midpoint of the segment
            mx = int((pt1[0] + pt2[0]) / 2)
            my = int((pt1[1] + pt2[1]) / 2)

            # Distance from midpoint to contour
            dist = cv2.pointPolygonTest(main_contour, (mx, my), True)
            if abs(dist) > max_distance_to_contour:
                continue  # Skip if farther than 100 pixels

            angle = math.degrees(math.atan2(dy, dx))
            cx = (pt1[0] + pt2[0]) / 2.0
            cy = (pt1[1] + pt2[1]) / 2.0

            try:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

                pts = np.array([[pt1], [pt2]], dtype=np.float32)
                pts_rot = cv2.transform(pts, M)
                x1r, y1r = pts_rot[0][0]
                x2r, y2r = pts_rot[1][0]

                x_min = int(min(x1r, x2r))
                x_max = int(max(x1r, x2r))
                y_center = int((y1r + y2r) / 2)
                y_min = max(0, y_center - line_width // 2)
                y_max = min(rotated.shape[0], y_center + line_width // 2)

                if x_min >= x_max or y_min >= y_max:
                    continue
                roi = rotated[y_min:y_max, x_min:x_max]
                if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue

                # Export ROI image
                cv2.imwrite(f"roi_segment_{i}.png", roi)

                # Export line overlay image
                line_img = rotated.copy()
                cv2.line(line_img, (int(x1r), int(y1r)), (int(x2r), int(y2r)), (0, 255, 0), 2)
                cv2.imwrite(f"line_segment_{i}.png", line_img)

                # Enhanced preprocessing
                roi_up = FloodFillApp.preprocess_roi_for_ocr(roi)
                
                cv2.imwrite(f"roi_up_segment_{i}.png", roi_up)
                # Try different Tesseract configs
                configs = [
                    #"--psm 7",  # Treat as a single text line
                    "--psm 6",  # Assume a block of text
                    #"--psm 11"  # Sparse text
                ]
                ocr_texts = []
                for config in configs:
                    ocr_texts.append(pytesseract.image_to_string(roi, config=config).strip())
                    #roi_up_180 = cv2.rotate(roi_up, cv2.ROTATE_180)
                    #ocr_texts.append(pytesseract.image_to_string(roi_up_180, config=config).strip())

                # Only add OCR results that match either regex pattern, or all if both are blank
                for text in ocr_texts:
                    if not text:
                        continue  # Skip empty OCR results

                    # Accept all if both regexes are blank
                    if not direction_pattern_str and not distance_pattern_str:
                        include_candidate = True
                    else:
                        include_candidate = any([
                            direction_pattern and direction_pattern.match(text),
                            distance_pattern and distance_pattern.match(text)
                        ])

                    if include_candidate:
                        self.ocr_candidate_boxes.append((cx, cy, x_max - x_min, y_max - y_min, angle))
                        ocr_results.append((i+1, text, (pt1, pt2)))

            except Exception as e:
                print(f"Exception on line {i}: {e}")
                continue

        # Draw rotated blue rectangles for OCR candidate boxes
        for cx, cy, w, h, angle in self.ocr_candidate_boxes:
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.polylines(img, [box], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue    

        self.table_text.delete("1.0", tk.END)
        if ocr_results:
            for idx, text, (pt1, pt2) in ocr_results:
                candidate = next(
                    ((cx, cy, w, h, angle) for cx, cy, w, h, angle in self.ocr_candidate_boxes
                     if abs(cx - (pt1[0] + pt2[0]) / 2.0) < 1 and abs(cy - (pt1[1] + pt2[1]) / 2.0) < 1),
                    None
                )
                if candidate:
                    cx, cy, w, h, angle = candidate
                    self.table_text.insert(
                        tk.END,
                        f"Line {idx}: {text} (from {pt1} to {pt2}) | Center: ({int(cx)}, {int(cy)})\n"
                    )
                else:
                    self.table_text.insert(
                        tk.END,
                        f"Line {idx}: {text} (from {pt1} to {pt2})\n"
                    )
        else:
            self.table_text.insert(tk.END, "No valid text found along decimated contour lines.")

        # Show overlays in preview (use img, not highlight_img)
        preview = self.viewport.get_view(img)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        

    

class Viewport:
    def __init__(self, image_width, image_height, canvas_width, canvas_height):
        self.image_width = image_width
        self.image_height = image_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def get_view(self, image):
        view_w = int(self.canvas_width / self.zoom)
        view_h = int(self.canvas_height / self.zoom)

        x1 = int(self.offset_x)
        y1 = int(self.offset_y)
        x2 = x1 + view_w
        y2 = y1 + view_h

        # Create a blank canvas (white background)
        canvas = np.ones((view_h, view_w, 3), dtype=np.uint8) * 255

        # Compute image region to copy
        img_x1 = max(0, x1)
        img_y1 = max(0, y1)
        img_x2 = min(x2, self.image_width)
        img_y2 = min(y2, self.image_height)

        # Compute where to paste it on the canvas
        paste_x1 = max(0, -x1)
        paste_y1 = max(0, -y1)
        paste_x2 = paste_x1 + (img_x2 - img_x1)
        paste_y2 = paste_y1 + (img_y2 - img_y1)

        # Copy image region into canvas
        canvas[paste_y1:paste_y2, paste_x1:paste_x2] = image[img_y1:img_y2, img_x1:img_x2]

        # Resize to canvas size
        resized = cv2.resize(canvas, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_AREA)
        return resized


if __name__ == "__main__":
    root = tk.Tk()
    app = FloodFillApp(root)
    root.mainloop()

