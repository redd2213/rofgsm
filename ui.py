import customtkinter as ctk
from tkinter import filedialog
import threading
from PIL import Image
from engine import load_image, fgsm_attack, pgd_attack, save_image, get_prediction, clip_attack, get_clip_distribution
from utils import validate_image_path, generate_output_path

# Color scheme
BG_PRIMARY = "#0a0a0a"
BG_SECONDARY = "#111111"
BG_PANEL = "#1a1a1a"
BORDER = "#2a2a2a"
TEXT_PRIMARY = "#e0e0e0"
TEXT_DIM = "#666666"
GREEN = "#00ff88"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class TangleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Tangle")
        self.configure(fg_color=BG_PRIMARY)
        self.resizable(True, True)
        self.after(100, lambda: self.state('zoomed'))

        self.input_path = None
        self.output_path = None

        self._build_topbar()
        self._build_main()

    def _build_topbar(self):
        self.top_bar = ctk.CTkFrame(self, fg_color=BG_SECONDARY, height=60, corner_radius=0)
        self.top_bar.pack(fill="x", side="top")
        self.top_bar.pack_propagate(False)

        self.logo_label = ctk.CTkLabel(
            self.top_bar,
            text="TANGLE",
            font=ctk.CTkFont(family="Cascadia Code", size=24, weight="bold"),
            text_color=GREEN
        )
        self.logo_label.pack(side="left", padx=30, pady=15)

        self.tagline_label = ctk.CTkLabel(
            self.top_bar,
            text="AI Mimicry is not art.",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            text_color=TEXT_DIM
        )
        self.tagline_label.pack(side="left", padx=0, pady=15)

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self, fg_color=BG_PRIMARY, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)

        self._build_left()
        self._build_right()

    def _build_left(self):
        self.left_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_PRIMARY, corner_radius=0)
        self.left_frame.pack(fill="both", expand=True, side="left")

        self.preview_frame = ctk.CTkFrame(self.left_frame, fg_color=BG_PRIMARY, corner_radius=0)
        self.preview_frame.pack(fill="both", expand=True, side="top")

        # Original panel
        self.original_frame = ctk.CTkFrame(self.preview_frame, fg_color=BG_PANEL, corner_radius=8)
        self.original_frame.pack(fill="both", expand=True, side="left", padx=(20, 10), pady=(20, 10))

        self.original_title = ctk.CTkLabel(
            self.original_frame,
            text="ORIGINAL",
            font=ctk.CTkFont(family="Cascadia Code", size=13),
            text_color=TEXT_DIM
        )
        self.original_title.pack(side="top", pady=(15, 0))

        self.original_image_label = ctk.CTkLabel(self.original_frame, text="")
        self.original_image_label.pack(expand=True)

        self.select_btn = ctk.CTkButton(
            self.original_frame,
            text="+ SELECT IMAGE",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            fg_color=BG_SECONDARY,
            hover_color=BORDER,
            text_color=TEXT_PRIMARY,
            corner_radius=4,
            command=self.select_image
        )
        self.select_btn.pack(side="bottom", pady=15, padx=15, fill="x")

        # Protected panel
        self.protected_frame = ctk.CTkFrame(self.preview_frame, fg_color=BG_PANEL, corner_radius=8)
        self.protected_frame.pack(fill="both", expand=True, side="left", padx=(10, 20), pady=(20, 10))

        self.protected_title = ctk.CTkLabel(
            self.protected_frame,
            text="PROTECTED",
            font=ctk.CTkFont(family="Cascadia Code", size=13),
            text_color=GREEN
        )
        self.protected_title.pack(side="top", pady=(15, 0))

        self.protected_image_label = ctk.CTkLabel(self.protected_frame, text="")
        self.protected_image_label.pack(expand=True)

        self.save_btn = ctk.CTkButton(
            self.protected_frame,
            text="↓ SAVE IMAGE",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            fg_color=BG_SECONDARY,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            corner_radius=4,
            state="disabled",
            command=self.save_image
        )
        self.save_btn.pack(side="bottom", pady=15, padx=15, fill="x")

        # Terminal log
        self.log_frame = ctk.CTkFrame(self.left_frame, fg_color=BG_PANEL, height=150, corner_radius=8)
        self.log_frame.pack(fill="x", side="bottom", padx=20, pady=(0, 20))
        self.log_frame.pack_propagate(False)

        self.log_box = ctk.CTkTextbox(
            self.log_frame,
            fg_color=BG_PANEL,
            text_color=GREEN,
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            corner_radius=0,
            state="disabled"
        )
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_right(self):
        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_SECONDARY, width=280, corner_radius=0)
        self.right_frame.pack(fill="y", side="right", padx=0, pady=0)
        self.right_frame.pack_propagate(False)

        self.attack_label = ctk.CTkLabel(
            self.right_frame,
            text="ATTACK TYPE",
            font=ctk.CTkFont(family="Cascadia Code", size=13),
            text_color=TEXT_DIM
        )
        self.attack_label.pack(anchor="w", padx=25, pady=(30, 10))

        self.attack_var = ctk.StringVar(value="PGD")

        self.fgsm_btn = ctk.CTkRadioButton(
            self.right_frame,
            text="FGSM  —  fast",
            variable=self.attack_var,
            value="FGSM",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            text_color=TEXT_PRIMARY,
            fg_color=GREEN,
            border_color=BORDER
        )
        self.fgsm_btn.pack(anchor="w", padx=25, pady=5)

        self.pgd_btn = ctk.CTkRadioButton(
            self.right_frame,
            text="PGD  —  strong",
            variable=self.attack_var,
            value="PGD",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            text_color=TEXT_PRIMARY,
            fg_color=GREEN,
            border_color=BORDER
        )
        self.pgd_btn.pack(anchor="w", padx=25, pady=5)

        self.clip_btn = ctk.CTkRadioButton(
            self.right_frame,
            text="CLIP  —  style",
            variable=self.attack_var,
            value="CLIP",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            text_color=TEXT_PRIMARY,
            fg_color=GREEN,
            border_color=BORDER
        )
        self.clip_btn.pack(anchor="w", padx=25, pady=5)

        # Divider
        ctk.CTkFrame(self.right_frame, fg_color=BORDER, height=1, corner_radius=0).pack(
            fill="x", padx=25, pady=20
        )

        self.strength_label = ctk.CTkLabel(
            self.right_frame,
            text="PROTECTION STRENGTH",
            font=ctk.CTkFont(family="Cascadia Code", size=13),
            text_color=TEXT_DIM
        )
        self.strength_label.pack(anchor="w", padx=25, pady=(0, 10))

        self.epsilon_var = ctk.DoubleVar(value=0.05)

        self.strength_slider = ctk.CTkSlider(
            self.right_frame,
            from_=0.01,
            to=0.4,
            variable=self.epsilon_var,
            fg_color=BORDER,
            progress_color=GREEN,
            button_color=GREEN,
            button_hover_color=TEXT_PRIMARY
        )
        self.strength_slider.pack(fill="x", padx=25, pady=(0, 5))

        self.epsilon_display = ctk.CTkLabel(
            self.right_frame,
            text="0.05",
            font=ctk.CTkFont(family="Cascadia Code", size=13),
            text_color=GREEN
        )
        self.epsilon_display.pack(anchor="w", padx=25)
        self.strength_slider.configure(command=self.update_epsilon_display)

        # Divider
        ctk.CTkFrame(self.right_frame, fg_color=BORDER, height=1, corner_radius=0).pack(
            fill="x", padx=25, pady=20
        )

        self.protect_btn = ctk.CTkButton(
            self.right_frame,
            text="PROTECT",
            font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
            fg_color=GREEN,
            hover_color="#00cc66",
            text_color="#000000",
            corner_radius=4,
            height=45,
            state="disabled",
            command=self.run_protection
        )
        self.protect_btn.pack(fill="x", padx=25, pady=(0, 10))

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"> {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def update_epsilon_display(self, value):
        self.epsilon_display.configure(text=f"{float(value):.2f}")

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")]
        )
        if not path:
            return

        is_valid, message = validate_image_path(path)
        if not is_valid:
            self.log(f"Error: {message}")
            return

        self.input_path = path
        self.log(f"Loaded: {path.split('/')[-1]}")

        img = Image.open(path)
        img.thumbnail((500, 500))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.original_image_label.configure(image=ctk_img)
        self.original_image_label.image = ctk_img

        self.protect_btn.configure(state="normal")

    def run_protection(self):
        self.protect_btn.configure(state="disabled")
        self.log(f"Starting {self.attack_var.get()} attack...")
        thread = threading.Thread(target=self._protect_worker)
        thread.start()

    def _protect_worker(self):
        try:
            epsilon = self.epsilon_var.get()
            attack_type = self.attack_var.get()

            if attack_type == "CLIP":
                self.after(0, lambda: self.log("Analyzing style embedding..."))

                # Distribution before
                dist_before = get_clip_distribution(self.input_path)
                self.after(0, lambda: self.log("BEFORE:"))
                for label, score in dist_before:
                     self.after(0, lambda l=label, s=score: self.log(f"  {l}: {s}%"))

                perturbed, original_size = clip_attack(self.input_path, epsilon=epsilon)
                self.output_path = generate_output_path(self.input_path)
                save_image(perturbed, self.output_path)

                # Distribution after
                dist_after = get_clip_distribution(self.output_path)
                self.after(0, lambda: self.log("AFTER:"))
                for label, score in dist_after:
                    self.after(0, lambda l=label, s=score: self.log(f"  {l}: {s}%"))
            else:
                original_tensor, model_tensor, original_size = load_image(self.input_path)

                label_before, conf_before = get_prediction(model_tensor)
                self.after(0, lambda: self.log(f"Before: {label_before} ({conf_before}%)"))

                if attack_type == "PGD":
                    perturbed = pgd_attack(original_tensor, model_tensor, epsilon=epsilon)
                else:
                    perturbed = fgsm_attack(original_tensor, model_tensor, epsilon=epsilon)

                self.output_path = generate_output_path(self.input_path)
                save_image(perturbed, self.output_path)

                original_tensor2, model_tensor2, _ = load_image(self.output_path)
                label_after, conf_after = get_prediction(model_tensor2)
                self.after(0, lambda: self.log(f"After: {label_after} ({conf_after}%)"))

            self.after(0, self._on_protection_done)

        except Exception as e:
            self.after(0, lambda: self.log(f"Error: {str(e)}"))
            self.after(0, lambda: self.protect_btn.configure(state="normal"))

    def _on_protection_done(self):
        self.log(f"Protected. Saved as: {self.output_path.split('/')[-1]}")

        img = Image.open(self.output_path)
        img.thumbnail((500, 500))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.protected_image_label.configure(image=ctk_img)
        self.protected_image_label.image = ctk_img

        self.save_btn.configure(state="normal", text_color=TEXT_PRIMARY)
        self.protect_btn.configure(state="normal")

    def save_image(self):
        if not self.output_path:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if dest:
            Image.open(self.output_path).save(dest)
            self.log(f"Saved to: {dest.split('/')[-1]}")

app = TangleApp()
app.mainloop()