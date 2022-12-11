from MATPLOTRC import *
import sqlite3

class Plot_Summary:
    def __init__(self,database):
        self.conn=sqlite3.connect(database)
        self.curs=self.conn.cursor()
    
    def plot_test_rewards(self,date_times,agents):
        plt.figure()
        plt.xlabel('step')
        plt.ylabel('test_reward')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        for t,a in zip(date_times,agents):
            self.curs.execute('''select recordsize from recordvalue
                where date='%s' and algorithm='test_return' and recordname='%s'
                order by step'''%(t,a))
            rows=self.curs.fetchall()
            plt.plot(range(len(rows)),rows,label=a)
        plt.legend()
        plt.savefig('test_reward.jpg',bbox_inches='tight')
    
    def plot_latest_rewards(self,agents):
        assert len(agents)
        if len(agents)>1:
            self.curs.execute('''select max(date),recordname from recordvalue
                where algorithm='test_return' and recordname in %s
                group by recordname'''%str(tuple(agents)))
        else:
            self.curs.execute('''select max(date),recordname from recordvalue
            where algorithm='test_return' and recordname='%s'
            group by recordname'''%agents[0])
        rows=self.curs.fetchall()
        self.plot_test_rewards(*zip(*rows))

if __name__=='__main__':
    pls=Plot_Summary('record.db')
    pls.plot_latest_rewards(['td3','ac','sac','dqn','ppo'])

