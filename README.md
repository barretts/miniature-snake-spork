# Miniature Snake Spork

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

tkinter basic GUI for running CrisperWhisper on an audio file and generating an SRT subtitle

## Table of Contents

- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## Setup Instructions

### Prerequisites

Before proceeding with the installation, ensure that you have the following installed on your system:

- Python (version 3.6 or higher)
- pip (Python package installer)
- git
- other stuff I bet

### Installation

To install the required dependencies for this project, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/barretts/miniature-snake-spork.git
   cd miniature-snake-spork
   ```

2. **Create a virtual environment** (recommended):
   It is good practice to use a virtual environment to manage dependencies for different projects.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   The project uses `pip` and `requirements.txt` to manage its dependencies. Run the following command in your terminal:

   ```bash
   git clone https://github.com/nyrahealth/CrisperWhisper.git
   pip install -r CrisperWhisper\requirements.txt
   pip install -r requirements.txt
   ```

   On Windows you can run `setup.bat`

## Usage

To run the application, execute the main script or any specific module as needed. Here is a basic example:

```bash
python gui_app.py
```

## Contributing

Contributions are welcome! If you have any ideas for new features, improvements, or bug fixes, please feel free to open a pull request. For major changes, please first open an issue to discuss what you would like to change.
