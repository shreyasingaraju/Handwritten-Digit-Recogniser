import sys
from PyQt5.QtWidgets import QApplication

from modules.project_model import ModelWrapper
from modules.project_gui import ProjectGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())