# Assetto Corsa Gym: WSL2 Ubuntu + NVIDIA GPU Setup

This guide explains how to set up and run Assetto Corsa via Steam/Proton inside an **Ubuntu 24.04 WSL2** environment leveraging your host NVIDIA GPU.

---

## **0. Prerequisites & WSL2 GPU Acceleration**

Before proceeding, ensure your Windows host and WSL2 environment are configured for GUI and GPU passthrough:

1. **Host Driver**: Install the latest **NVIDIA Windows Graphics Driver** on your host Windows OS (do **NOT** install an NVIDIA display driver inside WSL2).
2. **WSL2 GUI (WSLg)**: Ensure you are on Windows 11 (or Windows 10 Build 19044+) so WSLg is enabled.
3. **Verify GPU in WSL2**:
Open your Ubuntu WSL2 terminal and check that NVIDIA CUDA/drivers are accessible:
```bash
nvidia-smi

```


4. **Install System Dependencies**:
```bash
sudo apt update && sudo apt install -y \
  build-essential \
  libgl1-mesa-dri \
  libgl1-mesa-glx \
  zenity \
  python3-pip \
  git \
  wget \
  curl

```



---

## **1. Install Steam & Proton Setup in WSL2**

### **1.1 Install Steam inside WSL2**

Install Steam directly within your Ubuntu environment:

```bash
sudo apt update
sudo apt install -y steam

```

Run Steam once to initialize:

```bash
steam &

```

### **1.2 Download & Run ProtonUp-Qt**

1. Download the AppImage:
```bash
wget https://github.com/DavidoTek/ProtonUp-Qt/releases/download/v2.10.0/ProtonUp-Qt-2.10.0-x86_64.AppImage
chmod +x ProtonUp-Qt*.AppImage
./ProtonUp-Qt*.AppImage

```


2. In ProtonUp-Qt:
* Select **Install for: Steam** (it will auto-detect `~/.local/share/Steam/compatibilitytools.d/`).
* Click **Add version** $\rightarrow$ **GE-Proton** $\rightarrow$ **GE-Proton9-2** $\rightarrow$ **Install**.


3. Restart Steam (`killall steam && steam &`).

### **1.3 Enable Proton for Assetto Corsa**

1. Open Steam inside WSL2.
2. Right-click **Assetto Corsa** $\rightarrow$ **Properties** $\rightarrow$ **Compatibility**.
3. Check **"Force the use of a specific Steam Play compatibility tool"** and choose **GE-Proton9-2**.

### **1.4 Install Required Windows Fonts into Proton Prefix**

Run the following script to supply the missing UI fonts required by Assetto Corsa:

```bash
FONT_DIR="$HOME/.local/share/Steam/steamapps/compatdata/244210/pfx/drive_c/windows/Fonts"
mkdir -p "$FONT_DIR"

wget -O "$FONT_DIR/Verdana.ttf" \
      https://raw.githubusercontent.com/matomo-org/travis-scripts/master/fonts/Verdana.ttf

wget -O "$FONT_DIR/segoeuiz.ttf" \
      https://raw.githubusercontent.com/xamarin/evolve-presentation-template/master/Fonts/Segoe%20UI/segoeuiz.ttf

wget -O "$FONT_DIR/segoeui.ttf" \
      https://raw.githubusercontent.com/xamarin/evolve-presentation-template/master/Fonts/Segoe%20UI/segoeui.ttf

wget -O "$FONT_DIR/verdanai.ttf" \
      https://raw.githubusercontent.com/dolbydu/font/master/Sans/Verdana/verdanai.ttf

```

### **1.5 Launch Assetto Corsa (First Run)**

Launch Assetto Corsa via Steam to initialize the Proton prefix (wait ~3–5 minutes). Once it loads to the main screen, exit the game.

> **Note**: If the game fails to start a race session, switch to **GE-Proton9-26** in ProtonUp-Qt.

---

## **2. Content Manager Setup**

### **2.1 Download Content Manager**

Download and extract Content Manager inside WSL2:

```bash
wget -O CM.zip https://acstuff.ru/app/latest
unzip CM.zip

```

### **2.2 Create Hard Link for Content Manager**

Navigate to the Assetto Corsa game directory:

```bash
AC_PATH="$HOME/.local/share/Steam/steamapps/common/assettocorsa"
cd "$AC_PATH"

# Backup original executable
mv AssettoCorsa.exe AssettoCorsaBackup.exe

# Move Content Manager into folder and create hard link
mv ~/Content\ Manager.exe "$AC_PATH/"
ln "Content Manager.exe" AssettoCorsa.exe

```

### **2.3 Launch Content Manager**

Launch Assetto Corsa via Steam. Because of the hard link, **Content Manager** will launch instead of the main game executable.

When prompted for the **Assetto Corsa root folder** in Content Manager settings, set:

```text
Z:\home\<your_wsl_username>\.local\share\Steam\steamapps\common\assettocorsa\

```

*(Note: Proton maps the Linux root `/` to drive `Z:\` in Windows/Wine path syntax).*

### **2.4 Install Custom Shaders Patch (CSP)**

1. Download CSP v0.2.1 from [https://acstuff.club/patch/?info=0.2.1](https://acstuff.club/patch/?info=0.2.1).
2. Drag and drop the downloaded `.zip` file into the active Content Manager GUI window to install.

### **2.5 Restore Original Game Executable Name**

Exit Content Manager and run:

```bash
cd "$AC_PATH"
mv AssettoCorsa.exe AssettoCorsa_content_manager.exe
mv AssettoCorsaBackup.exe AssettoCorsa.exe

```

---

## **3. Assetto Corsa Gym Plugin & Configuration Files**

### **3.1 Copy Plugin Files**

Copy the Python plugin to the Assetto Corsa apps folder:

```bash
AC_PATH="$HOME/.local/share/Steam/steamapps/common/assettocorsa"

mkdir -p "$AC_PATH/apps/python/"
cp -r assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par \
   "$AC_PATH/apps/python/sensor_par"

```

### **3.2 Copy Controller & System Libraries**

```bash
mkdir -p "$AC_PATH/cfg/controllers/savedsetups"
mkdir -p "$AC_PATH/system/x64"

cp assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/Vjoy_linux.ini \
   "$AC_PATH/cfg/controllers/savedsetups/"

cp assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/WASD.ini \
   "$AC_PATH/cfg/controllers/savedsetups/"

cp -r assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/DLLs/* \
   "$AC_PATH/system/x64/"

cp -r assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/Lib/* \
   "$AC_PATH/system/x64/"

```

---

## **4. Xbox Driver Setup (Virtual Inputs on Linux)**

WSL2 uses `xboxdrv` as a virtual controller interface to send steering/pedal actions to Assetto Corsa via D-Bus:

### **4.1 Install `xboxdrv` and D-Bus tools**

```bash
sudo apt update
sudo apt install -y xboxdrv dbus-x11

```

### **4.2 Start `xboxdrv` Daemon**

Before starting your RL client or training scripts, launch `xboxdrv` in a background terminal or session:

```bash
sudo xboxdrv --daemon --silent --mimic-xpad --type xbox360 --dbus disabled

```

---

## **5. Game Configuration Settings**

1. Launch Assetto Corsa via Steam.
2. **Enable Plugin**: Go to **Options > General > UI Modules** $\rightarrow$ Check **`sensor_par`**.
3. **Load Controls**: Go to **Options > Controls** $\rightarrow$ Select and load **vJoy Linux**.
4. **Video Settings**: Set **Frame Rate Limit** to **50 FPS** (`Options > Video > Display`).
5. **Assists Settings**:

| Setting | Recommended Value |
| --- | --- |
| **Automatic Gearbox** | **ON** |
| **Automatic Clutch** | **Enabled** |
| **Traction Control** | **OFF** |
| **Stability Control** | **OFF** |
| **ABS** | **OFF** |
| **Tyre Blankets** | **ON** |
| **Penalties** | **ON** |

---

## **6. Verification & Troubleshooting**

### **6.1 Verify Virtual Inputs**

Run the maps creator test notebook inside your Python environment:

```bash
jupyter notebook test_maps_creator.ipynb

```

If configured correctly, the output will return the steering response mapping graph and display:

```text
Closest Steering Command to reach 240.0°: 0.5342

```

### **6.2 Debugging Plugin Logs**

If the plugin fails or telemetry is missing, check the Proton Python logs:

```bash
cat "$HOME/.local/share/Steam/steamapps/compatdata/244210/pfx/drive_c/users/steamuser/Documents/Assetto Corsa/logs/py_log.txt"

```