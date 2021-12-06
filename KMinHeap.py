class kminheap:

    def __init__(self, k:int):
        self.elements=[]
        self.k=k

    def insert(self,value,data):
        self.elements.append((value,data))
        self.elements.sort(key=lambda e: e[0])
        if len(self.elements)>self.k:
            self.elements=self.elements[0:self.k]
        # if len(self.elements)==0:
        #     self.elements.append((value,data))
        #     return
        # if data<=(self.elements[0])[0]:
        #     self.elements.insert(0,(value,data))
        # if data>(self.elements[0])[0]
        #     self.elements.
        # for i in range(0,len(self.elements)-1):

