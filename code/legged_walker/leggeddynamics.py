""" dynamics of a legged walker robot"""

import numpy as np
import openmdao.api as om


class lockedKneeDynamics(om.Group):
    
    def initialize(self):
        self.options.declare('num_nodes')
        self.options.declare('g', default=9.81, desc='gravity constant')

    def setup(self):
        nn = self.options('num_nodes')

        # inputs
        self.add_input('L', shape=(1,),units='m',desc="length of one leg") 

        # length paramters
        self.add_input('a1',shape=(1,),units='m',desc='shank length below point mass')
        self.add_input('b1', shape=(1,),units='m', desc='shank length above point mass')

        self.add_input('a2', shape=(nn,),units='m', desc='thigh length below points mass')
        self.add_input('b2', shape=(nn,),units='m', desc='thigh length above point mass')

        # q1 and q2
        self.add_input('q1', shape=(1,),units='rad', desc='q1 angle')
        self.add_input('q2', shape=(1,),units='rad', desc='q2 angle')

        # masses
        self.add_input('m_H', shape=(1,),units='kg', desc='hip mass')
        self.add_input('m_t', shape=(1,),units='kg', desc='thigh mass')
        self.add_input('m_s', shape=(1,),units='kg', desc='shank mass')


        # outputs
        # angular accelerations of q1 and q1 (state rates)
        self.add_output('q1_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q1')
        self.add_output('q2_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q2')

        #partials
        """ 
        To do
        """

        def compute(self, inputs, outputs):
            g = self.options['g']
            L = inputs['L']
            a1 = inputs['a1']
            a2 = inputs['a2']
            b1 = inputs['b1']
            b2 = inputs['b2']
            m_H = inputs['m_H']
            m_t = inputs['m_t']
            m_s = inputs['m_s']
            q1 = inputs['q1']
            q2 = inputs['q2']

            ls = a1 + b1
            lt = a2 + b2

            # matrix components
            H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*(L**2)
            H12 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(q2-q1))
            H22 = m_t*b2**2 + m_s*(lt + b1)**2
            h = -(m_t*b2 + m_s*(lt + b1))*(L*np.sin(q1-q2))
            G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.sin(q1)
            G2 = (m_t*b2 + m_s*(lt + b1))*g*np.sin(q2)


