import sys
from PyQt5.QtWidgets import QApplication

from modules.projectmodel import ProjModel
from modules.projectgui import ProjectGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())

# TODO:
# - Restructure to make code adhere to MVC better
# - Rename variables, functions for clarity and consistency
# - New class for drawing box - hopefully this will solve cursor offset issues
# - DONE: Move saved images, models etc. into their own folder
# - DONE: Remove saved images that were saved for debug purposes
# - Fix "QCoreApplication::exec: The event loop is already running" error when clicking random
# - Remove image processing code from MouseReleaseEvent, move to model class instead
# - Add error dialogs instead of printing caught errors to terminal 
# - Make pen a circle
# - Test if sharpen filter actually does anything
# - Check if files are already downloaded?