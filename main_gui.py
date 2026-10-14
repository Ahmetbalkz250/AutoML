import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from data_processor import load_and_prepare_data, get_column_names
from models_classification import train_logistic_regression, train_svc
from models_regression import train_linear_regression, train_svr

class AutoMLApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
      # Main Window Settings
        self.title("AutoML Studio Pro")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- 1. TARGET COLUMN SELECTION (Dropdown) ---
        self.label_hedef = ctk.CTkLabel(self, text="Select Target Column:", font=("Arial", 22, "bold"))
        self.label_hedef.pack(pady=(60, 15))
        
        # Using OptionMenu instead of Entry
        self.combo_hedef = ctk.CTkOptionMenu(self, width=450, height=50, font=("Arial", 16), 
                                             values=["Please select a file first..."])
        self.combo_hedef.pack(pady=10)
        
        # --- 2. FILE SELECTION BUTTON ---
        self.btn_dosya_sec = ctk.CTkButton(self, text="📂 Select Data File (.csv / .xlsx)", 
                                           font=("Arial", 18, "bold"), width=350, height=60,
                                           fg_color="#27ae60", hover_color="#2ecc71", 
                                           command=self.dosya_sec)
        self.btn_dosya_sec.pack(pady=30)
        
        self.label_dosya_yolu = ctk.CTkLabel(self, text="Selected File: None", text_color="gray", font=("Arial", 14))
        self.label_dosya_yolu.pack(pady=10)
        
        # --- 3. MODELS (4 Buttons Grid) ---
        self.frame_modeller = ctk.CTkFrame(self)
        self.frame_modeller.pack(pady=40, padx=60, fill="both", expand=True)
        
        # Centering the columns
        self.frame_modeller.grid_columnconfigure(0, weight=1)
        self.frame_modeller.grid_columnconfigure(1, weight=1)
        
        # Classification Buttons (Blue)
        self.btn_logistic = ctk.CTkButton(self.frame_modeller, text="Train Logistic Regression", 
                                          font=("Arial", 18, "bold"), width=320, height=80,
                                          command=self.run_logistic)
        self.btn_logistic.grid(row=0, column=0, padx=30, pady=40)
        
        self.btn_svc = ctk.CTkButton(self.frame_modeller, text="Train SVC (Classification)", 
                                     font=("Arial", 18, "bold"), width=320, height=80,
                                     command=self.run_svc)
        self.btn_svc.grid(row=0, column=1, padx=30, pady=40)
        
        # Regression Buttons (Orange)
        self.btn_linear = ctk.CTkButton(self.frame_modeller, text="Train Linear Regression", 
                                        font=("Arial", 18, "bold"), width=320, height=80,
                                        fg_color="#d35400", hover_color="#e67e22", 
                                        command=self.run_linear)
        self.btn_linear.grid(row=1, column=0, padx=30, pady=40)
        
        self.btn_svr = ctk.CTkButton(self.frame_modeller, text="Train SVR (Regression)", 
                                     font=("Arial", 18, "bold"), width=320, height=80,
                                     fg_color="#d35400", hover_color="#e67e22", 
                                     command=self.run_svr)
        self.btn_svr.grid(row=1, column=1, padx=30, pady=40)
        
        self.secilen_dosya_yolu = None
    def dosya_sec(self):
        # Opens the standard OS file dialog
        dosya = filedialog.askopenfilename(
            title="Select Data File for Model",
            filetypes=(("Data Files", "*.csv *.xlsx *.xls"), ("All Files", "*.*"))
        )
        
        if dosya:
            self.secilen_dosya_yolu = dosya
            # Showing only the filename to the user, not the full path
            dosya_adi = os.path.basename(dosya)
            self.label_dosya_yolu.configure(text=f"Selected File: {dosya_adi}", text_color="white")

            basarili, kolonlar = get_column_names(dosya)
            if basarili and kolonlar:
                self.combo_hedef.configure(values=kolonlar) # Load columns into the menu
                self.combo_hedef.set(kolonlar[-1]) # The target column is usually at the end, select it automatically
            else:
                messagebox.showerror("Read Error", "Could not read the columns of the file! Please check the format.")
                self.combo_hedef.configure(values=["Error Occurred"])
                self.combo_hedef.set("Error Occurred")

    def run_logistic(self):
        # 1. SECURITY CHECKS: Is a file selected? Is a target column entered?
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
            
        hedef_sutun = self.combo_hedef.get() 
        # Önceki kodda İngilizceye çevirdiğimiz açılır menü uyarılarıyla birebir aynı olmalı ki hata vermesin:
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return

        # 2. DATA PROCESSING: Sending the file to data_processor and getting X and y
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)
        
        if not basarili_mi:
            # If the process fails, display the error message from the 'sonuc' variable and stop.
            messagebox.showerror("Data Error", sonuc)
            return
            
        # If successful, unpack X and y
        X, y = sonuc
        
        # 3. ENGINE IGNITION: Send X and y to the Logistic Regression engine
        try:
            skor, model, scaler = train_logistic_regression(X, y)
            
            # Saving the trained model and scaler to the program's memory (self)
            # to use later when predicting "new data".
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Classification"
            
            # 4. SCORE CHECK (The 80% threshold you requested)
            if skor >= 0.80:
                messagebox.showinfo("Model Success!", 
                                    f"Great! The model was trained with high accuracy.\n\nF1 Score: {skor:.4f}\n\nYou can now predict new data.")
            else:
                messagebox.showwarning("Low Performance", 
                                       f"Model trained, but the score is below 80%.\n\nF1 Score: {skor:.4f}\n\nTry uploading better data.")
                                       
        except Exception as e:
            # To prevent the program from crashing if a mathematical error occurs during training:
            messagebox.showerror("Engine Error", f"A critical error occurred while training the model:\n{str(e)}")
    def run_svc(self):
        # 1. SECURITY CHECKS
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        # Açılır menüdeki İngilizce uyarılarla eşleşmesi için burayı da güncelledik:
        if hedef_sutun == "Please select a file first..." or hedef_sutun == "Error Occurred":
            messagebox.showerror("Missing Information", "Please select a valid target column!")
            return
        
        # 2. DATA PREPARATION
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Data Error", sonuc)
            return
            
        X, y = sonuc

        # 3. ENGINE IGNITION
        try: 
            skor, model, scaler = train_svc(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "SVC"

            if skor >= 0.80:
                messagebox.showinfo("Model Success!", 
                                    f"Great! The model was trained with high accuracy.\n\nF1 Score: {skor:.4f}\n\nYou can now predict new data.")
            else:
                messagebox.showwarning("Low Performance", 
                                       f"Model trained, but the score is below 80%.\n\nF1 Score: {skor:.4f}\n\nTry uploading better data.")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"A critical error occurred while training the model:\n{str(e)}")
    def run_linear(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Eksik Bilgi", "Lütfen önce bir veri dosyası (.csv/.xlsx) seçin!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        if hedef_sutun == "Önce Dosya Seçin..." or hedef_sutun == "Hata Oluştu":
            messagebox.showerror("Eksik Bilgi", "Lütfen geçerli bir hedef sütun seçin!")
            return
        
        # 2. VERİ HAZIRLIĞI
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Veri Hatası", sonuc)
            return
            
        X, y = sonuc

        # 3. MOTORU ATEŞLEME
        try: 
            skor, model, scaler = train_linear_regression(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Linear Regresyon"

            if skor >= 0.80:
                messagebox.showinfo("Model Başarılı!", 
                                    f"Harika! Model yüksek başarıyla eğitildi.\n\nR2 Skoru: {skor:.4f}\n\nArtık yeni verileri tahmin edebilirsiniz.")
            else:
                messagebox.showwarning("Düşük Başarı", 
                                       f"Model eğitildi ancak skor %80'in altında kaldı.\n\nR2 Skoru: {skor:.4f}\n\nDaha iyi veriler yüklemeyi deneyin.")
                                       
        except Exception as e:
            messagebox.showerror("Motor Hatası", f"Model eğitilirken kritik bir hata oluştu:\n{str(e)}")

    def run_svr(self):
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Eksik Bilgi", "Lütfen önce bir veri dosyası (.csv/.xlsx) seçin!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
        if hedef_sutun == "Önce Dosya Seçin..." or hedef_sutun == "Hata Oluştu":
            messagebox.showerror("Eksik Bilgi", "Lütfen geçerli bir hedef sütun seçin!")
            return
        
        # 2. VERİ HAZIRLIĞI
        basarili_mi, sonuc = load_and_prepare_data(self.secilen_dosya_yolu, hedef_sutun)       
        if not basarili_mi:
            messagebox.showerror("Veri Hatası", sonuc)
            return
            
        X, y = sonuc

        # 3. MOTORU ATEŞLEME
        try: 
            skor, model, scaler = train_svr(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "SVR"

            if skor >= 0.80:
                messagebox.showinfo("Model Başarılı!", 
                                    f"Harika! Model yüksek başarıyla eğitildi.\n\nR2 Skoru: {skor:.4f}\n\nArtık yeni verileri tahmin edebilirsiniz.")
            else:
                messagebox.showwarning("Düşük Başarı", 
                                       f"Model eğitildi ancak skor %80'in altında kaldı.\n\nR2 Skoru: {skor:.4f}\n\nDaha iyi veriler yüklemeyi deneyin.")
                                       
        except Exception as e:
            messagebox.showerror("Motor Hatası", f"Model eğitilirken kritik bir hata oluştu:\n{str(e)}")

if __name__ == "__main__":
    app = AutoMLApp()
    app.mainloop()