import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from data_processor import load_and_prepare_data, get_column_names
from models_classification import train_logistic_regression, train_svc, train_gaussienbayes
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
            # Artık skor değil, 'metrikler' sözlüğü geliyor
            metrikler, model, scaler = train_logistic_regression(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Classification"
            
            # Paketleri açıyoruz
            f1 = metrikler["F1"]
            auc = metrikler["AUC"]
            cm = metrikler["CM"]
            
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Model yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Model eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

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
        # 3. MOTORU ATEŞLEME (Sınıflandırma Örneği)
        try:
            # Artık skor değil, 'metrikler' sözlüğü geliyor
            metrikler, model, scaler = train_svc(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Classification"
            
            # Paketleri açıyoruz
            f1 = metrikler["F1"]
            auc = metrikler["AUC"]
            cm = metrikler["CM"]
            
            mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
            
            if f1 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Model yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Model eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_linear(self):
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
            metrikler, model, scaler = train_linear_regression(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Linear Regression"

            r2 = metrikler["R2"]
            mae = metrikler["MAE"]
            mse = metrikler["MSE"]
            
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"

            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Model yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Model eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_svr(self):
        # 1. SECURITY CHECKS
        if not self.secilen_dosya_yolu:
            messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
            return
        
        hedef_sutun = self.combo_hedef.get() 
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
            metrikler, model, scaler = train_svr(X, y)
            
            self.aktif_model = model
            self.aktif_scaler = scaler
            self.aktif_model_tipi = "Linear Regression"

            r2 = metrikler["R2"]
            mae = metrikler["MAE"]
            mse = metrikler["MSE"]
            
            mesaj = f"R2 Skoru: {r2:.4f}\nOrtalama Mutlak Hata (MAE): {mae:.4f}\nOrtalama Kare Hata (MSE): {mse:.4f}"

            if r2 >= 0.80:
                messagebox.showinfo("Model Başarılı!", f"Harika! Model yüksek başarıyla eğitildi.\n\n{mesaj}")
            else:
                messagebox.showwarning("Düşük Başarı", f"Model eğitildi ancak skor düşük.\n\n{mesaj}")
                                       
        except Exception as e:
            messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")

    def run_gaussienNB(self):
        # 1. SECURITY CHECKS
                if not self.secilen_dosya_yolu:
                    messagebox.showerror("Missing Information", "Please select a data file (.csv/.xlsx) first!")
                    return
                
                hedef_sutun = self.combo_hedef.get() 
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
                    metrikler, model, scaler = train_gaussienbayes(X,y)
                    
                    self.aktif_model = model
                    self.aktif_scaler = scaler
                    self.aktif_model_tipi = "GaussienNB"
        
                    f1 = metrikler["F1"]
                    auc = metrikler["AUC"]
                    cm = metrikler["CM"]
                    
                    mesaj = f"F1 Skoru: {f1:.4f}\nAUC: {auc}\n\nKarmaşıklık Matrisi (Confusion Matrix):\n{cm}"
                                
                    if f1 >= 0.80:
                        messagebox.showinfo("Model Başarılı!", f"Harika! Model yüksek başarıyla eğitildi.\n\n{mesaj}")
                    else:
                        messagebox.showwarning("Düşük Başarı", f"Model eğitildi ancak skor düşük.\n\n{mesaj}")
                                                           
                except Exception as e:
                    messagebox.showerror("Engine Error", f"Hata:\n{str(e)}")
       
                

if __name__ == "__main__":
    app = AutoMLApp()
    app.mainloop()