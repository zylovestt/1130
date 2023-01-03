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
    
    def plot_latest(self,agents):
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
        self.plot_loss(*zip(*rows))
    
    def plot_loss(self,date_times,agents):
        for t,a in zip(date_times,agents):
            self.curs.execute('''select distinct recordname from recordvalue
                where date='%s' and algorithm='%s'
                '''%(t,a))
            rows=self.curs.fetchall()
            for r,*_ in rows:
                self.curs.execute('''select recordsize from recordvalue
                    where date='%s' and algorithm='%s' and recordname='%s'
                    order by step'''%(t,a,r))
                values=self.curs.fetchall()
                plt.figure()
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.plot(range(len(values)),values,label=a+'_'+r)
                plt.legend()
                plt.savefig(a+'_'+r,bbox_inches='tight')


if __name__=='__main__':
    pls=Plot_Summary('record.db')
    pls.plot_latest(['td3','ac','sac','dqn','ppo','td3sac'])
    # pls.plot_latest(['td3sac'])