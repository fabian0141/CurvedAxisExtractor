from extractor.wall import Wall


class CircleWall(Wall):

    THRESHOLD = 2

    def __init__(self, circle, state):
        self.type = Wall.CIRCLE
        self.circle = circle
        self.state = state