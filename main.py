import sys
from PyQt5.QtWidgets import QApplication

from modules.project_model import ModelWrapper
from modules.project_gui import ProjectGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProjectGUI()
    sys.exit(app.exec_())

# TODO:
# - Restructure to make code adhere to MVC better
# - Rename variables, functions for clarity and consistency 
#   PEP 8 is a style guide for python, would improve readability if we followed it
#       - Variables should be named with underscores e.g. variable
#       - According to PEP 8 methods and functions should be named with underscores but 
#         due to PyQT using CamelCase, we should use CamelCase for our own methods and functions for consistency
#       - Classes should be in CamelCase 
# - New class for drawing box - hopefully this will solve cursor offset issues
# - DONE: Move saved images, models etc. into their own folder
# - DONE: Remove saved images that were saved for debug purposes
# - Fix "QCoreApplication::exec: The event loop is already running" error when clicking random
# - Remove image processing code from MouseReleaseEvent, move to model class instead
# - Add error dialogs instead of printing caught errors to terminal 
# - DONE: Make pen a circle
# - Test if sharpen filter actually does anything
# - Check if files are already downloaded?
# - DONE: Clear the existing plotted graph if "random" is clicked
# - DONE: Make the loaded image / digit appear in the pixmap in the lower left instead of the small graph
# - Change randomClicked so that the image is saved in the correct aspect ratio
# - Move the positions of the main window and plot window so that plot doesn't cover the buttons of main