from tkinter import *
from PIL import Image, ImageTk
import threading

class GridWorld:
	def __init__(self, size, environment, main_function):
		self.root = Tk()
		self.root.resizable(height=None, width=None)
		self.root.title("Comm-DMAR")

		self.main = main_function
		self.env = environment

		self.padding = 20
		self.size = size

		self.colors=[]
		self.agent_images = []
		self.grid_labels = []
		for i in range(100):
			self.colors+=[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,
						19,20,21,22,23,24,25,26,27,28,29,30,31]
		self.lone_color_ind = 10

		self.collect_agent_images()
		self.create_grid_labels()
		self.update_static_grid_labels()
		self.update_agent_init_labels()

		self.create_buttons()

		# blankLabel=Label(self.root, text="     ", borderwidth=6,
		# 				padx=self.padding,pady=self.padding,relief="solid")

	def run_visualizer(self):
		self.root.mainloop()

	def create_buttons(self):
		goButton=Button(self.root,text='Start',pady=10,padx=50,command=self.start)
		goButton.grid(row=self.size+1, column=0, columnspan=self.size-4)
		quitButton=Button(self.root, text="Quit", command=self.root.destroy)
		quitButton.grid(row=self.size+2, column=0, columnspan=self.size-4)

	def start(self):
		t=threading.Thread(target=self.main)
		t.start()

	def collect_agent_images(self):
		for i in range(31):
			img= Image.open('images/agent'+str(i+1)+'.png')
			img = img.resize((50, 50), Image.LANCZOS)
			img=ImageTk.PhotoImage(img)
			self.agent_images.append(img)

	def create_grid_labels(self):
		for i in range(self.size):
			row_labels = []
			for j in range(self.size):
				row_labels.append(Label(self.root, text="     ", borderwidth=6,
					bg='#333366', padx=self.padding,pady=self.padding,relief="solid"))
				row_labels[j].grid(row=i,column=j)
			self.grid_labels.append(row_labels)

	def update_static_grid_labels(self):
		vertices = list(self.env.grid_graph.nodes)
		for (i,j) in vertices:
			self.grid_labels[i][j].grid_forget()
			self.grid_labels[i][j] = Label(self.root, text="     ", borderwidth=6,
					padx=self.padding,pady=self.padding,relief="solid")
			self.grid_labels[i][j].grid(row=i,column=j)

		task_images = []
		for i in range(len(self.env.task_vertices)):
			img= Image.open('images/task.png')
			img = img.resize((50, 50), Image.ANTIALIAS)
			img=ImageTk.PhotoImage(img)
			task_images.append(img)
		# task_images = [img]*len(self.env.task_vertices)
		
		for i,task in enumerate(self.env.task_vertices):
			self.grid_labels[task[0]][task[1]].grid_forget()
			self.grid_labels[task[0]][task[1]] = Label(
			image=task_images[i],borderwidth=6, padx=6,pady=4.495,relief="solid")
			self.grid_labels[task[0]][task[1]].image = task_images[i]
			self.grid_labels[task[0]][task[1]].grid(row=task[0],column=task[1])

	def update_agent_init_labels(self):
		for (i,j) in self.env.agent_vertices:
			self.grid_labels[i][j].grid_forget()
			self.grid_labels[i][j] = Label(image=self.agent_images[10],borderwidth=6,
			padx=6,pady=4.495,relief="solid")
			self.grid_labels[i][j].image = self.agent_images[10]
			self.grid_labels[i][j].grid(row=i,column=j)

	def change_cell(self, x, y, cell_type, agent_color=10):
		self.grid_labels[x][y].grid_forget()
		if cell_type == 'agent':
			self.grid_labels[x][y] = Label(image=self.agent_images[agent_color],
				borderwidth=6,padx=6,pady=4.495,relief="solid")
			self.grid_labels[x][y].image = self.agent_images[agent_color]
			self.grid_labels[x][y].grid(row=x,column=y)
		elif cell_type == 'blank':
			self.grid_labels[x][y].grid_forget()
			self.grid_labels[x][y] = Label(self.root, text="     ", borderwidth=6,
					padx=self.padding,pady=self.padding,relief="solid")
			self.grid_labels[x][y].grid(row=x,column=y)

