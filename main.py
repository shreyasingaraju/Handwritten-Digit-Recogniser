import sys
from PyQt5.QtWidgets import QApplication

from modules.ProjectModel import ProjModel
from modules.ProjectGUI import ProjectGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())