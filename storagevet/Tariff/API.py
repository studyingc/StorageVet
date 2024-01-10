"""
Copyright (c) 2023, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import requests
import pprint
import Period as period
import EnergyTier as et
import csv

USERS_OPENEI_API_KEY = ''
ADDRESS = '3420 Hillview Ave, Palo Alto, CA 94304'
LIMIT = 20

class API:
    def __init__(self):
        self.URL = "https://api.openei.org/utility_rates"
        # OpenEI (Open Energy Information) API를 사용하여 유틸리티 요금 정보를 가져오는 클래스를 초기화하는 부분
        self.PARAMS = {'version': 5, 'api_key': USERS_OPENEI_API_KEY, 'format': 'json',
                       'address': ADDRESS, 'limit': LIMIT}
        # API 요청에 사용될 매개변수를 정의하는 딕셔너리
        self.r = requests.get(url=self.URL, params=self.PARAMS)
        # requests 라이브러리를 사용하여 OpenEI API에 GET 요청을 보냄
        self.data = self.r.json()
        # API 응답을 JSON 형식으로 파싱하여 데이터를 저장
        if 'error' in self.data.keys(): #응답 데이터에 'error' 키가 있는지 확인
            raise Exception(f'\nBad API call: {self.data["error"]}')
            # 'error' 키가 존재하면 API 호출에 오류가 있음을 나타내므로 예외를 발생
        self.tariff = None
        self.energyratestructure = []
        self.energyweekdayschedule = []
        self.energyweekendschedule = []
        self.schedule = []
        self.energy_period_list = []
        # 유틸리티 요금 및 기타 정보를 저장할 변수들을 초기화

        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None
        # 기타 변수들을 초기화
 
        self.weekday_date_list = []
        self.weekend_date_list = []
        self.date_list = []
        # 주중 및 주말 날짜 리스트를 초기화

    def print_all(self):
    # OpenEI API에서 검색된 모든 유틸리티 요금에 대한 정보를 출력
        """
        Prints necessary identifying information of all tariffs that show from result page on OpenEI

        """
        count = 1
        # 각 유틸리티 요금에 대한 번호를 나타내는 변수를 초기화
        for item in self.data["items"]: # self.data에 저장된 API 응답에서 "items" 키에 해당하는 리스트를 순회
            print("---------------------------------------------------", count) # 현재의 번호를 출력
            print("Utility.......", item["utility"])
            # 해당 유틸리티 요금의 이름을 출력
            print("Name..........", item["name"])
            # 유틸리티 요금의 이름을 출력
            if "enddate" in item:
                print("End Date......", item["enddate"])
                # 유틸리티 요금에 종료 날짜가 있는 경우, 종료 날짜를 출력
            if "startdate" in item:
                print("Start Date....", item["startdate"])
                # 유틸리티 요금에 시작 날짜가 있는 경우, 시작 날짜를 출력
            print("EIA ID........", item["eiaid"])
            # 유틸리티 요금의 EIA ID를 출력
            print("URL...........", item["uri"])
            # 유틸리티 요금의 URL을 출력
            if "description" in item:
                print("Description...", item["description"])
                # 유틸리티 요금에 설명이 있는 경우, 설명을 출력
            print(" ")
            count += 1
            # 다음 유틸리티 요금에 대한 번호를 증가

    def reset(self):
    #  reset이 호출될 때마다 특정 객체의 티어 값을 초기화하는 역할
        """
        Resets tariff's tier values to None; necessary for print_index

        """
        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None
        # 해당 객체의 티어 값들이 초기 상태로 돌아감

    def print_index(self, index):
    # 선택한 인덱스에 해당하는 요금제의 에너지 요금 구조를 설정
        """
        Establishes all periods and tiers of the tariff using period and tier objects

        Args:
            index (Int): user input for which tariff they choose

        """
        i = index
        while i not in range(1, LIMIT + 1):
            print('That index is out of range, please try another...')
            i = int(input("Which tariff would you like to use?..."))
            # 사용자로부터 선택한 인덱스가 유효한 범위 내에 있는지 확인하고, 
            # 유효하지 않은 경우 적절한 인덱스를 다시 입력하도록 유도하는 코드
        label = self.data["items"][i - 1]["label"]
        params = {'version': 5, 'api_key': USERS_OPENEI_API_KEY, 'format': 'json', 'getpage': label, 'detail': 'full'}
        r = requests.get(url=self.URL, params=params)
        self.tariff = r.json()
        # 선택한 인덱스에 해당하는 요금제의 정보를 OpenEI API를 통해 가져오기 위해 API 호출을 수행하고, 해당 정보를 self.tariff에 저장

        if "energyratestructure" in self.tariff["items"][0]: # 만약 선택한 요금제에 energyratestructure 정보가 있다면
            # print(self.tariff["items"][0]["energyratestructure"])
            self.energyratestructure = self.tariff["items"][0]["energyratestructure"]
            # 해당 정보를 self.energyratestructure에 저장
            pcount = 1  # period count
            tcount = 1  # tier count
            for p in self.energyratestructure:
                self.energy_period_list.append(period.Period(pcount))
                for i in p:
                    if "max" in i:
                        self.max = i["max"]

                    if "rate" in i:
                        self.rate = i["rate"]

                    if "unit" in i:
                        self.unit = i["unit"]

                    if "adjustment" in i:
                        self.adj = i["adjustment"]

                    if "sell" in i:
                        self.sell = i["sell"]

                    self.energy_period_list[pcount - 1].add(et.Tier(tcount, self.max, self.rate, self.unit, self.adj, self.sell))
                    # 각 기간과 티어의 정보를 반복하여 self.energy_period_list에 설정
                    tcount += 1
                    self.reset()
                    #  reset 메서드를 호출하여 기존의 값을 초기화하여 다음 기간 및 티어 정보를 처리할 수 있도록 함
                tcount = 1
                pcount += 1

    def print_energy_structure(self):
    # 전력 요금 구조와 각 구간이 언제 활성화되는지의 일정을 터미널에 출력
        """
        Prints energy structure, month and hour schedule of when every period is active, to terminal

        """
        pprint.pprint(self.tariff)
        # pprint 모듈을 사용하여 self.tariff에 저장된 전력 요금 구조를 예쁘게 출력
        if not self.energy_period_list:  # if list is empty it is not printed
            pass
            # self.energy_period_list가 비어 있으면, 구간 정보를 출력하지 않고 함수를 종료
        else:
            print(" ")
            print("Tiered Energy Usage Charge Structure")
            # 구간 정보를 출력
            for period in self.energy_period_list:
                print(" ")
                period.tostring()
                # period 객체에 정의된 tostring 메서드를 호출하여 해당 구간의 세부 정보를 출력
                for tier in period.tier_list:
                    tier.tostring()
                    # 구간 내의 각 티어에 대해 반복하면서 티어의 정보를 출력
            print(" ")
            # 각 구간에 대해 반복하면서 구간의 정보를 출력

        #  self.tariff에서 추출한 주중 및 주말의 전력 사용 스케줄을 가공하여 터미널에 출력하는 부분
        self.energyweekdayschedule = self.tariff["items"][0]["energyweekdayschedule"]
        # self.tariff에서 주중의 전력 사용 스케줄을 추출하여 self.energyweekdayschedule 변수에 저장
        self.energyweekendschedule = self.tariff["items"][0]["energyweekendschedule"]
        # self.tariff에서 주말의 전력 사용 스케줄을 추출하여 self.energyweekendschedule 변수에 저장
        for year in self.energyweekdayschedule: # 주중의 각 해에 대해 반복
            count = 0
            for month in year: # 주중의 해당 월에 대해 반복
                year[count] = month + 1 # 각 월의 값에 1을 더하여 출력 형식을 조정
                count += 1
            print(year)
        print('=----------------------------------------------------------------------=')
        for year in self.energyweekendschedule: # 주말의 각 해에 대해 반복
            count = 0
            for month in year:
                year[count] = month + 1
                count += 1
            print(year)
            # 위와 동일한 프로세스를 수행하여 주말의 전력 사용 스케줄을 출력
            # 주중과 주말의 전력 사용 스케줄을 연도별로 가공하여 터미널에 출력
       # 사용자는 전력 요금 구조와 각 구간의 세부 정보, 그리고 전력 사용의 계절적 변동성을 쉽게 확인할 수 있음
 
    def dates(self, dates, weekday):
    # energyweekdayschedule 또는 energyweekendschedule을 기반으로 하여 전력 사용 스케줄을 분석
    # 각 기간의 시작 월과 끝 월을 추출하는 역할
    # 주어진 dates 리스트에 각 기간의 정보를 저장
        """
        Looks at energy weekday schedule and establishes a list of periods to describe the schedule using start and end months

        """
        if weekday is True: # weekday가 True인 경우 
            schedule = self.energyweekdayschedule # energyweekdayschedule을 사용
        else: # False인 경우 
            schedule = self.energyweekendschedule # energyweekendschedule을 사용
        dates = dates
        switch = False  # true if period in row for the first time (start is set)
        # switch 변수를 초기화하고, 이 변수는 각 기간의 시작이 설정되었는지 여부
        period = 1      # period we are looking for in schedule 스케줄에서 찾을 기간을 나타내는 변수를 초기화
        month = 1       # current month of for loop  현재 월을 나타내는 변수를 초기화
        start = 0       # start month 시작 월을 나타내는 변수를 초기화
        end = 0         # end month 끝 월을 나타내는 변수를 초기화
        index = 0       # index to compare current row to next row 현재 행을 다음 행과 비교하기 위한 인덱스를 초기화

        # continues to loop unless period is not found in row
        while True:
            for row in schedule: # 스케줄의 각 행에 대해 반복
                # don't need to check past row 11
                if index != 11:
                    index += 1

                # switch is False if start has not been set
                if switch is False: # 시작이 설정되지 않은 경우
                    if period in row: # 현재 기간이 행에 있는 경우
                        start = month # 시작 월을 현재 월로 설정
                        end = month # 시작 월을 현재 월로 설정
                        switch = True 
                        month += 1
                    else:
                        month += 1

                # switch is True if start has been set
                elif switch is True: # 시작이 설정된 경우
                    if period in row: # 현재 기간이 행에 있는 경우
                        end = month # 끝 월을 현재 월로 설정
                        month += 1
                    else:
                        dates.append([period, start, end, period]) # 현재 기간의 정보를 dates 리스트에 추가
                        switch = False # switch를 False로 설정
                        month += 1

                # if for loop is on last loop, append what it has
                if month >= 13: 현재 월이 13보다 큰 경우(12월을 넘어가는 경우)
                    # if period was never given months it is not appended to list
                    if start == 0: 시작이 설정되지 않은 경우
                        period += 1
                        month = 1
                        index = 0
                        break
                    dates.append([period, start, end, period])
                    # 현재 기간의 정보를 dates 리스트에 추가
                    period += 1
                    month = 1
                    index = 0

                # if the next row is different from the current row it will append the current period and start a new one
                if schedule[index] != row: # 다음 행이 현재 행과 다른 경우
                    if start == 0: # 시작이 설정되지 않은 경우
                        continue
                    else:
                        dates.append([period, start, end, period])
                        # 현재 기간의 정보를 dates 리스트에 추가
                        switch = False

            # if start is 0 then the period does not exist in the schedule and we are done
            if period > len(self.energy_period_list): # 현재 기간이 energy_period_list의 길이보다 큰 경우
                self.rates(dates) # dates를 사용하여 rates 메서드를 호출
                break
            else: # 현재 기간이 energy_period_list의 길이보다 크지 않은 경우
                start = 0
                switch = False

    def hours(self, dates, weekday):
    # energyweekdayschedule 또는 energyweekendschedule을 기반으로 하여 각 기간의 활동 시간 범위를 결정
    # dates 리스트의 각 기간에 대해 활동 시간 정보를 추가
        """
        Looks at energy weekday schedule and establishes range of hours that a period is active for

        """
        if weekday is True:
            schedule = self.energyweekdayschedule
        else:
            schedule = self.energyweekendschedule
        dates = dates
        switch = 0
        start = 0     # start month
        end = 0       # end month
        ex_start = 0  # excluding start month
        ex_end = 0    # excluding end month
        time = 0

        for p in dates: # dates 리스트의 각 기간에 대해 반복
            period = p[3]   # period that we are finding active times for 현재 기간을 나타내는 변수를 설정
            index = p[1]-1  # look at starting month to find times that period is active 시작 월을 찾아볼 인덱스를 설정
            month = schedule[index] # 해당 인덱스에 대한 월을 설
            for hour in month: # 월의 각 시간에 대해 반복
                time += 1
                # case 0: start month has not yet been found, once found goes to case 1
                if switch == 0: # 상태가 0인 경우 (아직 시작 월을 찾지 않은 경우)
                    if hour == period: # 현재 시간이 기간과 같은 경우
                        start = time # 시작 시간을 현재 시간으로 설정
                        end = time # 종료 시간을 현재 시간으로 설정
                        switch = 1
                    else:
                        continue

                # case 1: start month is set, if hour is equal to period there is possible gap which goes to case 2
                elif switch == 1: # 상태가 1인 경우 (시작 시간이 설정된 경우)
                    if hour == period: # 현재 시간이 기간과 같은 경우
                        end = time # 종료 시간을 현재 시간으로 설정
                    else:
                        if start == 1: # 시작 시간이 1인 경우
                            continue
                        else:
                            ex_start = end # 제외 시작 시간을 종료 시간으로 설정
                            switch = 2 # 상태를 2로 설정

                # case 2: if there is a gap between a period, sets ex_end goes to case 3
                elif switch == 2: # 상태가 2인 경우 (기간 사이에 갭이 있는 경우)
                    if hour == period: # 현재 시간이 기간과 같은 경우
                        end = time # 종료 시간을 현재 시간으로 설정
                        ex_end = time # 제외 종료 시간을 현재 시간으로 설정
                        switch = 3 # 상태를 3으로 설정
                    else:
                        continue

                # case 3: sets end of gap 
                elif switch == 3: # 상태가 3인 경우
                    if hour == period: # 현재 시간이 기간과 같은 경우
                        end = time # 종료 시간을 현재 시간으로 설정
                    else:
                        continue

            if ex_end == 0: # 제외 종료 시간이 0인 경우,
                ex_start = None # 제외 시작 시간을 None으로 설정
                ex_end = None # 제외 종료 시간을 None으로 설정
            p.append(start-1) # 시작 시간을 dates 리스트에 추가
            p.append(end-1) # 종료 시간을 dates 리스트에 추가
            p.append(ex_start) # 제외 시작 시간을 dates 리스트에 추가
            p.append(ex_end) # 제외 종료 시간을 dates 리스트에 추가
            del p[3] # 기존의 3번 인덱스를 삭제
            ex_start = 0 # 제외 시작 시간을 초기화
            ex_end = 0 # 제외 종료 시간을 초기화
            switch = False # switch 변수를 False로 설정
            time = 0 # time = 0: 시간을 초기화

    def rates(self, dates):
    # 각 에너지 기간에 대해 요금을 할당
    # 각 에너지 기간에 대한 최고 요금을 계산하고, 해당 요금을 dates 리스트에 추가
        """
        Assigns rates to each energy period in date_list before list is formatted

        """
        temp_list = []
        #  임시 리스트를 초기화
        for p in self.energy_period_list: #  에너지 기간 리스트를 반복
            p.get_highest_rate() # 현재 에너지 기간에 대한 최고 요금을 계산
            temp_list.append(p.highest_rate) # 최고 요금을 temp_list에 추가

        for period in dates: # 리스트의 각 기간에 대해 반복
            period.append(temp_list[period[3]-1]) 
            # 해당 기간에 대한 최고 요금을 dates 리스트에 추가
            # period[3]은 기간을 나타냄, 최고 요금은 temp_list에서 가져옴 

    def clean_list(self, dates):
    # dates 리스트에서 중복 항목을 제거하고 각 기간의 시작 월에 따라 정렬
        """
        Removes duplicates from date_list and orders it according to starting month of every period

        """
        self.remove_duplicates(dates)
        # 중복 항목을 제거하는 메서드를 호출
        dates.sort(key=self.take_second)  # sorts list based on second element (starting month)
        # dates 리스트를 두 번째 요소(시작 월)를 기준으로 정렬
        count = 1
        for p in dates: # dates 리스트의 각 항목에 대해 반복
            p[0] = count # 항목의 첫 번째 요소(기간 번호)를 카운터로 설정
            count += 1 #  카운터를 증가
            p.append(p.pop(3))  # moves rate to end of list
            # 리스트에서 세 번째 항목(최고 요금)을 빼내어 맨 뒤에 추가 

    def remove_duplicates(self, dates):
    # 중복 항목을 제거하는 메서드
        """
        Removes all duplicates from a list leaving only one of an element

        """
        for p in dates: # dates 리스트의 각 항목에 대해 반복
            while dates.count(p) >= 2: #  현재 항목 p가 리스트에 두 번 이상 나타날 때까지 반복
                dates.remove(p) # 중복된 항목을 리스트에서 제거, 중복 항목을 하나만 남김

    def take_second(self, elem):
    # 주어진 리스트에서 두 번째 요소를 반환하는 함수
    # elem이라는 리스트를 인자로 받음
        """
        Args:
            elem: list

        Returns:
            elem[1] (Element): second element of list

        """
        return elem[1]
        # elem[1]을 반환, 이는 리스트의 두 번째 요소

    def run(self):
        self.print_all()
        # OpenEI API에서 얻은 전체 전력 요금 정보를 출력
        i = int(input("Which tariff would you like to use?..."))
        # 사용자에게 어떤 전력 요금을 사용할지 물어봄
        self.print_index(i)
        # 선택한 전력 요금에 대한 자세한 정보를 출력
        self.print_energy_structure()
        # 에너지 구조 및 각 기간이 활동하는 월 및 시간 스케줄을 출력

        print("WEEKDAY")
        weekday = []
        self.dates(weekday, True) # 에너지 요금 구조에서 활성 기간을 찾아서 이를 날짜 목록에 추가
        self.hours(weekday, True) # 활성 기간에 대한 시간 범위를 찾아서 날짜 목록에 추가
        self.clean_list(weekday) # 중복을 제거하고 시작 월에 따라 날짜 목록을 정렬
        for p in weekday:
            print(p)

        print("WEEKEND")
        weekend = []
        self.dates(weekend, False)
        self.hours(weekend, False)
        self.clean_list(weekend)
        for p in weekend:
            print(p)

        # 주중과 주말에 대해 각각 dates(), hours(), clean_list()를 실행하고 결과를 출력

        """
            with open('tariff.csv', mode='w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['Billing Period', 'Start Month', 'End Month', 'Start Time', 'End Time', 'Excluding Start Time', 'Excluding End Time', 'Weekday?', 'Value', 'Charge', 'Name_optional'])
            for p in self.date_list:
                filewriter.writerow([p[0], p[1], p[2], p[3], p[4], p[5], p[6], None, p[7]])
            # 주석을 해제하면 CSV 파일로 결과를 저장하는 부분
        """


def main():
    api = API() # API 클래스의 인스턴스를 생성
    api.run() # run() 메서드를 호출하여 전체 프로그램을 실행

if __name__ == "__main__": main() 
#  구문은 현재 스크립트가 직접 실행될 때만 main() 함수를 호출하도록 하는 일반적인 Python 스크립트 구조
#  구문은 현재 스크립트가 직접 실행될 때만 main() 함수를 호출하도록 하는 일반적인 Python 스크립트 구조
