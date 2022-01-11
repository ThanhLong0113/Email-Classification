from main import * 
import tkinter as tk
from tkinter.constants import END, RIGHT, WORD, Y
from tkinter.filedialog import askopenfilename

# Create window
root = tk.Tk()
root.geometry("640x400")
root.title("Spam Email Filtering")
root.resizable(0, 0)
root.configure(bg="#404040")
scrollbar = tk.Scrollbar(root)

# Browse Function
def browsefunc():
    filename = askopenfilename(filetypes=(("txt files", "*.txt"), ))
    ent1.delete(0, tk.END)
    ent1.insert(0, filename)
    content = open(filename)
    ent2.configure(state="normal")
    ent2.delete('1.0', END)
    ent2.insert("1.0", content.read())
    ent2.configure(state="disabled")

# CheckSpam function
def checkSpam(content,path):
    if len(content) <= 1:
        popup = tk.Toplevel()
        popup.geometry("300x80")
        popup.wm_title("No file")
        label = tk.Label(
            popup,
            text="No file"
        )
        label.pack()
        button = tk.Button(
            popup,
            text="Close",
            font=40,
            command=lambda: popup.destroy()
        )
        button.pack()
    else:
        Spam = detectSpam(path)
        print(Spam)
        if Spam:
            popup = tk.Toplevel()
            popup.geometry("300x80")
            popup.wm_title("Result")
            label = tk.Label(
                popup,
                text="This is spam mail!")
            label.pack()
            button = tk.Button(
                popup,
                bg="#7f7f7f",
                text="Close",
                font=40,
                command=lambda: popup.destroy()
            )
            button.pack()
        else:
            popup = tk.Toplevel()
            popup.geometry("300x80")
            popup.wm_title("Result")
            label = tk.Label(
                popup,
                text="This isn't spam mail!"
            )
            label.pack()  
            button = tk.Button(
                popup,
                bg="#7f7f7f",
                text="Close",
                font=40,
                command=lambda: popup.destroy())
            button.pack()


label1 = tk.Label(
    root,
    text="Mail URL",
    bg="#7f7f7f"
)
label1.pack()

ent1 = tk.Entry(
    root,
    font=20,
    width=60,
    bg="#7f7f7f"
)
ent1.pack()

b1 = tk.Button(
    root,
    text="Choose file",
    height=1,
    font=20,
    bg="#7f7f7f",
    command=browsefunc
)
b1.pack()

label2 = tk.Label(
    root,
    bg="#7f7f7f",
    text="File content"
)
label2.pack()

ent2 = tk.Text(
    root,
    font=15,
    width=60,
    height=9,
    bg="#7f7f7f",
    yscrollcommand=scrollbar.set
)
scrollbar.config(
    command=ent2.yview
)
scrollbar.pack(
    side=RIGHT,
    fill=Y
)
ent2.pack()

b2 = tk.Button(
    root,
    text="Check",
    font=20,
    height=1,
    bg="#7f7f7f",
    command=lambda: checkSpam(ent2.get("1.0", END),ent1.get())
)
b2.pack()

root.mainloop()
