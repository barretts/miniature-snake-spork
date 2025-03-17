@echo off
if not exist "CrisperWhisper" (
    echo Cloning CrispWhisper repository...
    git clone https://github.com/nyrahealth/CrisperWhisper.git
)
echo Installing CrispWhisper dependencies...
pip install -r CrisperWhisper\requirements.txt
echo Installing project dependencies...
pip install -r requirements.txt
echo Setup complete.
pause
