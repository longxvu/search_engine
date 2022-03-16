class Posting:
    def __init__(self):
        self.doc_id = -1
        self.term_freq = 0
        self.positions = []

    def init_from_args(self, doc_id, positions):
        self.doc_id = doc_id
        self.term_freq = len(positions)
        self.positions = positions

    def parse_from_str(self, input_strs: str, parse_position: bool):
        if parse_position:
            input_strs_lst = input_strs.splitlines()
            self.doc_id, self.term_freq = map(int, input_strs_lst[0].split())   # doc_id term_freq 1st line
            self.positions = list(map(int, input_strs_lst[1].split()))  # positions 2nd line
        else:
            self.doc_id, self.term_freq = map(int, input_strs.split())

    def update_position_list(self, position):
        self.positions.append(position)
        self.term_freq += 1

    # return approximate size in bytes
    def get_approximate_size(self):
        # experiments showed that approx. size is ~ 3 times total number of items. Don't ask me why
        return 3 * (2 + self.term_freq)

    def positions_string(self):
        return " ".join(map(str, self.positions))

    def posting_string(self):
        return str(self.doc_id) + " " + str(self.term_freq)

    def __str__(self):
        return self.posting_string() + "\n" + self.positions_string()


# Testing purpose only
if __name__ == "__main__":
    posting = Posting()
    posting.init_from_args(3, [1, 2, 3, 5, 6])

    print(posting.get_approximate_size())
    print(str(posting))