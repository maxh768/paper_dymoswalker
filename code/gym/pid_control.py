class PD:
    def __init__(self, kp, kd, goal):
        self.kp = kp
        self.kd = kd
        self.goal = goal
        self.last_error = 0

    def observe(self, x):
        error = self.goal - x
        d_error = error - self.last_error
        self.last_error = error
        return self.kp * error + self.kd * d_error
    
class Controller:
    def __init__(self):
        self.cart = PD(kp=1, kd=100, goal=0)
        self.pole = PD(kp=5, kd=100, goal=0)

    def observe(self, cart_position, pole_angle):
        u_cart = self.cart.observe(cart_position)
        u_pole = self.pole.observe(pole_angle)
        if u_pole+u_cart > 0:
            action = 1
        elif u_pole+u_cart <0:
            action = -1
        else:
            action = 0
        return action