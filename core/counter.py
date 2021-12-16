from collections import deque


class CounterMeter:
    def __init__(self, window_size=None):
        self._total = 0
        self._count = 0
        self.window_size = window_size
        self.ratio = -1

    @property
    def total(self):
        return self._total

    @property
    def count(self):
        return self._count

    def update(self, sign):
        self.ratio = -1
        self._total += 1
        if sign:
            self._count += 1
        if self.window_size is not None and self._total == self.window_size:
            # TODO
            self.ratio = self._count / self._total
            self.clear()

    def clear(self):
        self._total = 0
        self._count = 0

    # @property
    # def ratio(self):
    #     return self._count / self._total


class KaBa:
    """
    example:
    self.start_job()
    for i in range(x):
        self.update(boxes)
    self.end_job()
    result = self.status
    """

    def __init__(self, area) -> None:
        self.start = False
        self.personCnt = CounterMeter(window_size=250)
        self.moneyCnt = CounterMeter()
        self.caseCnt = CounterMeter()
        # TODO
        self.area = area
        self.status = None

    def start_job(self):
        self.start = True

    def end_job(self):
        self.start = False

    def update(self, person_boxes, money_boxes, case_boxes):
        if not self.start:
            return
        self.personCnt.update(len(person_boxes) < 2)
        self.moneyCnt.update(len(money_boxes) >= 1)
        self.caseCnt.update(len(case_boxes) >= 1)

        if self.personCnt.ratio > 0.1:
            self.status = False
        else:
            self.status = (self.moneyCnt.count) > 50 and (self.caseCnt.count > 50)


class DoublePerson:
    """
    example:
    for i in range(x):
        self.update(boxes)
    result = self.status
    """
    def __init__(self, area) -> None:
        self.start = False
        self.end = False
        self.personCnt = CounterMeter()
        self.startCnt = CounterMeter(window_size=25)
        self.endCnt = CounterMeter(window_size=600)
        self.status = None
        self.area = area

    def start_job(self, num_person):
        self.startCnt.update(num_person > 1)
        if self.startCnt.ratio == 1:
            self.start = True
            self.startCnt.clear()

    def end_job(self, num_person):
        self.endCnt.update(num_person == 0)
        if self.endCnt.ratio == 1:
            self.end = True
            self.endCnt.clear()

    def update(self, head_boxes, person_boxes):
        num_person = (head_boxes and person_boxes) > 0.7  # box_iou
        if not self.start:
            self.start_job()
        else:
            self.personCnt.update(num_person >= 2)
            self.end_job()
        if self.end:
            self.status = self.personCnt.count / (self.total - self.endCnt.window_size) > 0.65

class CounterAction:
    """
    example:
    for i in range(x):
        self.update(boxes)
    result = self.status
    """
    def __init__(self, area) -> None:
        self.area = area
        self.sleepCnt = CounterMeter(window_size=500)
        self.playphoneCnt = CounterMeter(window_size=150)
        self.status = {'sleep': None, 'playphone': None}

    def update(self, sleep_boxes, playphone_boxes, phone_boxes):
        num_sleep = len(sleep_boxes)
        num_playphone = (playphone_boxes and phone_boxes) > 0
        self.sleepCnt.update(num_sleep >= 1)
        self.playphoneCnt.update(num_playphone >= 1)
        if self.sleepCnt.ratio >= 0.35:
            self.status['sleep'] = True
        if self.playphoneCnt.ratio >= 0.35:
            self.status['playphone'] = True

class CounterTransaction:
    def __init__(self, area) -> None:
        self.cashCnt = CounterMeter(10)
        self.personCnt = CounterMeter(10)
        self.status = {'cash': None, 'person': None}
        self.area = area

    def update(self, cash_boxes, person_boxes):
        self.cashCnt.update(len(cash_boxes) > 0)
        num_person = (person_boxes and person_boxes) < 0.95
        self.personCnt.update(num_person >= 2)
        if self.cashCnt.ratio == 1:
            self.status['cash'] = True
        if self.personCnt.ratio == 1:
            self.status['person'] = True

class CounterOutTransaction:
    """除去人脸识别"""
    def __init__(self) -> None:
        self.personCnt = CounterMeter()
        self.status = None

    def update(self, person_boxes, face_boxes):
        face_boxes = face_boxes.center in self.area
        if (face_boxes and face_boxes) > 0.05:
            return
        person_boxes = person_boxes and face_boxes
        if (person_boxes and person_boxes) > 0.3:
            return
        person_boxes = min(person_boxes.x - self.area.center)
        self.personCnt.update(len(person_boxes) >= 1)
        if self.personCnt.count >= 10:
            self.stauts = True 

class BussinessAnalysis:
    """除去人脸检测"""
    def __init__(self, area) -> None:
        self.area = area
        self.ids = {}
        self.current_id = None
        self.report_id = None

    def check_ids(self, id):
        if self.current_id is None:
            self.current_id = id
        elif id == self.current_id:
            pass
        else:
            self.report_id = self.current_id
            self.report()
            self.current_id = id

    def report(self):
        print(f"id {self.report_id} finished.")

    def update(self, person_tracks, face_boxes):
        face_boxes = face_boxes.center in self.area
        if (face_boxes and face_boxes) > 0.05:
            return
        person_tracks = person_tracks and face_boxes
        if (person_tracks and person_tracks) > 0.3:
            return
        person_tracks = min(person_tracks.x - self.area.center)
        track_id = person_tracks.id
        if track_id not in self.ids:
            self.ids[track_id] = CounterMeter(window_size=50)
        else:
            if self.ids[track_id].ratio == 1:
                self.check_ids(track_id)



if __name__ == "__main__":
    a = deque(maxlen=1)
    print(len(a))
