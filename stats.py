import csv


class Stats(object):

    def __init__(self, csv_file_name):
        self.num_games = 1
        self.time_step = 0
        self.csv_file_name = csv_file_name
        self.total_game_loss = 0

        self.csv_file = open(self.csv_file_name, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow((
            'game_number',
            'average_loss',
            'game_score',
            'game_duration'
        ))

        self.csv_file.flush()

    def log_time_step(self, loss):
        self.total_game_loss += loss
        self.time_step += 1

    def log_game(self, game_score, game_duration):

        assert self.time_step > 0

        average_loss = self.total_game_loss / self.time_step

        self.csv_writer.writerow((
            self.num_games,
            average_loss,
            game_score,
            game_duration
        ))

        self.num_games += 1
        self.time_step = 0
        self.total_game_loss = 0

        self.csv_file.flush()

    def close(self):
        self.csv_file.close()



