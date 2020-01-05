# -*- coding: utf-8 -*-
"""
Created on Fri Jan 2 15:47:39 2018

@author: MONSTER
"""

import sys
from PyQt4 import QtGui

from main import MainWindow

def main():
    app = QtGui.QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    return app.exec_()

if __name__ == "__main__":
    main()
