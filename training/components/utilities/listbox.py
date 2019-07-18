from tkinter import Listbox, Frame, Tk, Button
import os


class GetFileSelection:
    """Shows a listbox, where user can make selections from list.
    Returns list from given path with only the selected objects from directory,
    Obtained list is saved in global variable 'file_list'.

    Can be used to select samples to be processed in the scripts.
    """

    def __init__(self, list_path):
        self.master = Tk()
        self.master.title('Select samples to be processed.')
        frame = Frame(self.master)
        frame.pack()

        self.list_box = Listbox(frame, width=40, height=30, selectmode='extended')
        self.list_box.pack(side='left')

        files = self.insert_list(list_path)

        self.quit = Button(frame, text='Exit and run program', command=self.exit(files), width=15)
        self.quit.pack(side='bottom')

        self.button = Button(frame, text='Show list', command=self.print_list, width=15)
        self.button.pack(side='bottom')

        self.master.mainloop()

    def insert_list(self, list_path):
        """Inserts list elements to GUI"""
        files = os.listdir(list_path)
        files.sort()
        for k in range(len(files)):
            self.list_box.insert('end', files[k])
        return files

    def print_list(self):
        """Prints selected list on command line."""
        val = self.list_box.curselection()
        print('Selected values: {0}'.format(val))

    def exit(self, files):
        """Exits the GUI. File list is saved in a global variable."""
        global file_list
        file_list = self.list_box.curselection()
        files = [files[i] for i in file_list]
        file_list = files[:]
        print('')
        self.master.destroy()


if __name__ == '__main__':
    # Pipeline variables
    path = r"Y:\3DHistoData\Subvolumes_Insaf"

    # Run application
    GetFileSelection(path)
    print(file_list)
