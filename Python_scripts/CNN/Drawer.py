import matplotlib.pyplot as plt


class SliceDrawer:
    def __init__(self, img, projection='Z'):
        self.volume = img
        self.projection = projection
        self.fig, self.ax = plt.subplots()
        self.ax.volume = self.volume
        self.ax.index = 0

    def _select_projection(self, index):
        if self.projection == 'X':
            return self.volume[index, :, :]
        elif self.projection == 'Y':
            return self.volume[:, index, :]
        return self.volume[:, :, index]

    def draw(self):
        cax = self.ax.imshow(self._select_projection(self.ax.index), cmap='inferno')
        cbar = self.fig.colorbar(cax)
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.ax.set_title(f'Current slice {self.ax.index}')
        plt.show()

    def _show(self):
        self.ax.images[0].set_array(self._select_projection(self.ax.index))
        self.ax.set_title(f'Current slice {self.ax.index}')

    def next_slice(self):
        self.ax.index = (self.ax.index + 1) % self.volume.shape[2]
        self._show()

    def previous_slice(self):
        self.ax.index = (self.ax.index - 1) % self.volume.shape[2]
        self._show()

    def process_key(self, event):
        self.fig = event.canvas.figure
        self.ax = self.fig.axes[0]
        if event.key == 'j':
            self.next_slice()
        elif event.key == 'k':
            self.previous_slice()
        self.fig.canvas.draw()
