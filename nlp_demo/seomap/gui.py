from tkinter import *
import webbrowser

class MainWindow:
    def __init__(self):
        self.app = Tk()
        self.app.title("KeywordMap Login")
        self.app.geometry("300x250")
        self.label = Label(self.app, text="KeywordMap vBeta.1")
        self.label.place(x=110, y=40)
        self.login = Button(self.app, text="Log-in",
                            pady=5, padx=35, command=login)
        self.login.place(x=100, y=100)
        self.register = Button(self.app, text="Get a License",
                               pady=5, padx=10, command=register)
        self.register.place(x=100, y=150)

    def run(self):
        self.app.mainloop()


def login():
    pass


def register():
    url = "https://www.google.com"
    new = 1
    webbrowser.open(url,new=new)

app = MainWindow()
app.run()