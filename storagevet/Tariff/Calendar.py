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
import xlsxwriter
import os, sys
import pandas as pd

USERS_OPENEI_API_KEY = ''
ADDRESS = '3420 Hillview Ave, Palo Alto, CA 94304'
LIMIT = 20

class API:
    def __init__(self):
    # 클래스가 인스턴스화될 때 호출되는 메서드로, 해당 클래스의 객체를 초기화
        self.URL = "https://api.openei.org/utility_rates"
        self.PARAMS = {'version': 5, 'api_key': USERS_OPENEI_API_KEY, 'format': 'json',
                       'address': ADDRESS, 'limit': LIMIT}
        self.r = requests.get(url=self.URL, params=self.PARAMS)
        self.data = self.r.json()
        if 'error' in self.data.keys():
            raise Exception(f'\nBad API call: {self.data["error"]}')
        self.tariff = None
        self.energyratestructure = []
        self.energyweekdayschedule = []
        self.energyweekendschedule = []
        self.energy_period_list = []

        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None

        self.weekday_date_list = []
        self.weekend_date_list = []
        self.date_list = []

    def print_all(self):
    # OpenEI API에서 가져온 모든 요금에 대한 정보를 출력하는 메서드
        """
        Prints necessary identifying information of all tariffs that show from result page on OpenEI

        """
        count = 1
        for item in self.data["items"]:
            print("---------------------------------------------------", count)
            print("Utility.......", item["utility"]) # 유틸리티 이름 (utility)
            print("Name..........", item["name"]) # 요금 이름 (name)
            if "enddate" in item:
                print("End Date......", item["enddate"]) # 종료 날짜 (enddate) - 해당 정보가 존재하는 경우에만 출력
            if "startdate" in item:
                print("Start Date....", item["startdate"]) # 시작 날짜 (startdate) - 해당 정보가 존재하는 경우에만 출력
            print("EIA ID........", item["eiaid"]) # EIA ID (eiaid)
            print("URL...........", item["uri"]) # 요금의 URI (uri)
            if "description" in item:
                print("Description...", item["description"]) # 설명 (description) - 해당 정보가 존재하는 경우에만 출력
            print(" ")
            count += 1
            # 요금에 대한 정보를 구분하기 위해 count 변수를 사용하여 번호를 부여하고, 
            # 각 항목은 print 함수를 사용하여 화면에 출력

    def reset(self):
    # Tariff 클래스의 인스턴스에서 사용되며, 요금 객체의 티어 값을 초기화
    # print_index 메서드에서 각 티어의 값을 설정하기 전에 필요한 작업
        """
        Resets tariff's tier values to None; necessary for print_index

        """
        self.max = None
        self.rate = None
        self.unit = None
        self.adj = None
        self.sell = None
        # max, rate, unit, adj, sell을 모두 None으로 재설정하여 초기화

    def print_index(self, index):
    # 사용자가 선택한 특정 요금을 기반으로 기간과 티어 정보를 설정
    # 사용자가 선택한 인덱스를 받아서 해당 인덱스에 해당하는 요금의 세부 정보를 OpenEI API로부터 가져옴
    # 가져온 정보에는 해당 요금의 energyratestructure가 있음
        """
        Establishes all periods and tiers of the tariff using period and tier objects

        Args:
            index (Int): user input for which tariff they choose

        """
        i = index
        while i not in range(1, LIMIT + 1): # 선택한 인덱스 i가 1부터 LIMIT까지의 범위에 속하지 않을 때까지 반복
            print('That index is out of range, please try another...')
            # 선택한 인덱스가 범위를 벗어날 경우 사용자에게 메시지를 출력
            i = int(input("Which tariff would you like to use?..."))
            # 새로운 인덱스를 입력하도록 사용자에게 요청
        label = self.data["items"][i - 1]["label"]
        # 선택한 인덱스에 해당하는 OpenEI API 응답 데이터에서 요금의 라벨 정보를 가져옴
        params = {'version': 5, 'api_key': USERS_OPENEI_API_KEY, 'format': 'json', 'getpage': label, 'detail': 'full'}
        #  OpenEI API에 보낼 요청 파라미터를 설정, getpage에는 선택한 요금의 라벨이 들어감
        r = requests.get(url=self.URL, params=params)
        # 설정된 파라미터로 OpenEI API에 GET 요청을 보냄
        self.tariff = r.json()
        # API 응답을 JSON 형식으로 변환

        if "energyratestructure" in self.tariff["items"][0]: # API 응답 데이터에 "energyratestructure" 키가 있는지 확인
            # print(self.tariff["items"][0]["energyratestructure"])
            self.energyratestructure = self.tariff["items"][0]["energyratestructure"]
            # "energyratestructure"의 값을 가져와 self.energyratestructure에 저장
            pcount = 1  # period count
            # 기간의 수를 나타내는 변수를 초기화
            tcount = 1  # tier count
            # 티어의 수를 나타내는 변수를 초기화
            for p in self.energyratestructure: #  energyratestructure의 각 항목을 순회
                self.energy_period_list.append(period.Period(pcount))
                # 새로운 기간을 나타내는 Period 객체를 생성하고, energy_period_list에 추가
                for i in p:  energyratestructure의 각 항목을 순회
                    if "max" in i: # 현재 항목에 "max" 키가 있는지 확인
                        self.max = i["max"] # max" 키가 있다면 해당 값을 가져와 self.max에 저장

                    if "rate" in i: # 현재 항목에 "rate" 키가 있는지 확인
                        self.rate = i["rate"] # "rate" 키가 있다면 해당 값을 가져와 self.rate에 저장

                    if "unit" in i: # 현재 항목에 "unit" 키가 있는지 확인
                        self.unit = i["unit"] # "unit" 키가 있다면 해당 값을 가져와 self.unit에 저장

                    if "adjustment" in i: # 현재 항목에 "adjustment" 키가 있는지 확인
                        self.adj = i["adjustment"] # "adjustment" 키가 있다면 해당 값을 가져와 self.adj에 저장

                    if "sell" in i: # 현재 항목에 "sell" 키가 있는지 확인
                        self.sell = i["sell"] # "sell" 키가 있다면 해당 값을 가져와 self.sell에 저장

                    self.energy_period_list[pcount - 1].add(et.Tier(tcount, self.max, self.rate, self.unit, self.adj, self.sell))
                    # 현재 기간에 새로운 티어를 나타내는 Tier 객체를 생성, 해당 기간의 Period 객체에 추가
                    tcount += 1
                    # 티어의 수를 증가
                    self.reset()
                    # 티어 속성들을 초기화하는 reset 메서드를 호출
                tcount = 1 # 다음 기간을 위해 티어의 수를 초기화
                pcount += 1 # 다음 기간을 위해 기간의 수를 증가

    def print_energy_structure(self):
    # 해당 tariff가 가지고 있는 기간 및 티어 정보, 주중 및 주말에 기간이 활성화되는 월 및 시간 일정을 출력
    # 에너지 구조
        """
        Prints energy structure, month and hour schedule of when every period is active, to terminal

        """
        pprint.pprint(self.tariff)
        # 현재의 tariff 정보를 보기 좋게 출력
        if not self.energy_period_list:  # if list is empty it is not printed energy_period_list가 비어있지 않은 경우에만 실행
            pass
        else:
            print(" ")
            print("Tiered Energy Usage Charge Structure")
            for period in self.energy_period_list: # energy_period_list의 각 기간에 대해 반복
                print(" ")
                period.tostring() # 현재 기간을 나타내는 Period 객체의 tostring 메서드를 호출하여 정보를 출력
                for tier in period.tier_list: # 현재 기간의 각 티어에 대해 반복
                    tier.tostring() # 현재 티어를 나타내는 Tier 객체의 tostring 메서드를 호출하여 정보를 출력
            print(" ")

        self.energyweekdayschedule = self.tariff["items"][0]["energyweekdayschedule"]
        # 주중에 기간이 활성화되는 월 및 시간 일정 정보를 가져옴
        self.energyweekendschedule = self.tariff["items"][0]["energyweekendschedule"]
        # 주말에 기간이 활성화되는 월 및 시간 일정 정보를 가져옴
        for year in self.energyweekdayschedule: # 주중 일정에 대해 연도별로 반복
            count = 0
            for month in year: #  각 월에 대해 반복
                year[count] = month + 1 # 월의 값을 1 증가시킨 후, 해당 연도의 리스트에 할당
                count += 1 # 인덱스를 증가
            print(year)
        print('=----------------------------------------------------------------------=')
        for year in self.energyweekendschedule: # 주말 일정에 대해 연도별로 반복
            count = 0
            for month in year: #  각 월에 대해 반복
                year[count] = month + 1 # 월의 값을 1 증가시킨 후, 해당 연도의 리스트에 할당
                count += 1
            print(year)

    def calendar(self):
    # 엑셀 파일을 생성하고, 그 안에 주중 스케줄, 주말 스케줄 및 각 기간의 요금을 포함하는 세 개의 시트를 만듬
    # 셀 시트에는 시간과 월에 따른 색상으로 구분된 시간대별 기간 스케줄이 포함
    # 기간별로 다른 색상을 가진 조건부 서식을 적용하여 엑셀 시트를 시각적으로 구분
        """
        Makes an excel file with three spreadsheets: weekday schedule, weekend schedule, and the rates of each period

        """
        # create three workbook with three worksheets
        workbook = xlsxwriter.Workbook('calendar.xlsx')
        wksht_weekday = workbook.add_worksheet(name="Weekday")
        wksht_weekend = workbook.add_worksheet(name="Weekend")
        wksht_rates = workbook.add_worksheet(name="Rates")

        hours = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # conditional formatting that changes the format of a cell based on number value
        # yellow
        yellow = workbook.add_format() # 서식을 정의하기 위한 xlsxwriter의 add_format 메서드를 사용하여 생성된 객체
        yellow.set_align('center') # 셀 내용을 가운데 정렬하도록 설정
        yellow.set_bg_color('yellow') # 배경색을 노란색으로 설정
        yellow.set_bold() # 텍스트를 굵게 표시하도록 설정
        yellow.set_font_color('black') #  폰트 색상을 검정색으로 설정
        yellow.set_border(1) # 테두리를 추가
        yellow.set_border_color('white') # 테두리 색상을 흰색으로 설정
        cond_yellow = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 1, 'format': yellow})
        cond_yellow = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 1, 'format': yellow})
        # conditional_format 메서드를 사용하여 주중 및 주말 스케줄의 특정 셀 범위에 대한 조건부 서식을 정의
        # 셀 값이 1인 경우에 해당하는 셀에 yellow 서식을 적용
        # 작업은 주중과 주말 스케줄의 두 시트에 대해 각각 수행
        # blue
        blue = workbook.add_format()
        blue.set_align('center')
        blue.set_bg_color('blue')
        blue.set_bold()
        blue.set_font_color('white')
        blue.set_border(1)
        blue.set_border_color('white')
        cond_blue = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 2, 'format': blue})
        cond_blue = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 2, 'format': blue})

        # green
        green = workbook.add_format()
        green.set_align('center')
        green.set_bg_color('green')
        green.set_bold()
        green.set_font_color('white')
        green.set_border(1)
        green.set_border_color('white')
        cond_green = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 3, 'format': green})
        cond_green = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 3, 'format': green})

        # red
        red = workbook.add_format()
        red.set_align('center')
        red.set_bg_color('red')
        red.set_bold()
        red.set_font_color('black')
        red.set_border(1)
        red.set_border_color('white')
        cond_red = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 4, 'format': red})
        cond_red = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 4, 'format': red})

        # purple
        purple = workbook.add_format()
        purple.set_align('center')
        purple.set_bg_color('purple')
        purple.set_bold()
        purple.set_font_color('white')
        purple.set_border(1)
        purple.set_border_color('white')
        cond_purple = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 5, 'format': purple})
        cond_purple = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 5, 'format': purple})

        # lime
        lime = workbook.add_format()
        lime.set_align('center')
        lime.set_bg_color('lime')
        lime.set_bold()
        lime.set_font_color('black')
        lime.set_border(1)
        lime.set_border_color('white')
        cond_lime = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 6, 'format': lime})
        cond_lime = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '=', 'value': 6, 'format': lime})

        # else
        # 값이 6보다 큰 경우에 대한 기본 서식을 정의
        center = workbook.add_format() # 식을 정의하기 위한 xlsxwriter의 add_format 메서드를 사용하여 생성된 객체
        center.set_align('center') #  셀 내용을 가운데 정렬하도록 설정
        cond_else = wksht_weekday.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '>', 'value': 6, 'format': center})
        cond_else = wksht_weekend.conditional_format(1, 1, 12, 24, {'type': 'cell', 'criteria': '>', 'value': 6, 'format': center})

        # -------------------- weekday --------------------
        #  'Weekday' 스케줄의 엑셀 파일 워크시트에 시간 헤더를 작성하고 열 너비를 설정
        # write hours in header
        for i in range(len(hours)):
            wksht_weekday.write(0, i+1, hours[i])
        wksht_weekday.set_column(1, 24, 3.14, center)
        # hours 리스트에 있는 각 시간을 엑셀 헤더의 각 열에 작성하고, 그에 따른 열의 너비를 설정
        
        # 'Weekday' 스케줄의 엑셀 파일 워크시트에 월을 작성하고 월 열의 너비를 설정하는 부분입니다
        # write months in first column
        for i in range(len(months)):
            wksht_weekday.write(i+1, 0, months[i])
        wksht_weekday.set_column(0, 0, 4, center)
        # months 리스트에 있는 각 월을 엑셀의 첫 번째 열에 작성하고, 해당 열의 너비를 설정

        # write all periods conditional formatting in weekday schedule
        x = 0
        y = 0
        for month in self.energyweekdayschedule:
            for hour in month:
                if hour == 1:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_yellow)
                elif hour == 2:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_blue)
                elif hour == 3:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_green)
                elif hour == 4:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_red)
                elif hour == 5:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_purple)
                elif hour == 6:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_lime)
                else:
                    wksht_weekday.write(1 + y, 1 + x, hour, cond_else)
                x += 1
            x = 0
            y += 1
            #  'Weekday' 스케줄의 엑셀 파일 워크시트에 모든 기간에 대한 조건부 서식을 적용하여 스케줄을 작성하는 부분
            # 'energyweekdayschedule'에서 가져온 값이 특정 값에 해당하면 해당하는 조건부 포맷을 적용하여 엑셀에 해당 값을 작성
            # 예를 들어, hour가 1이면 cond_yellow 서식을 적용하여 해당 셀에 1을 작성
            # 모든 월과 시간에 대해 반복

        # -------------------- weekend --------------------
        # write hours in header
        for i in range(len(hours)):
            wksht_weekend.write(0, i+1, hours[i])
        wksht_weekend.set_column(1, 24, 3.14, center)

        # write months in first column
        for i in range(len(months)):
            wksht_weekend.write(i+1, 0, months[i])
        wksht_weekend.set_column(0, 0, 4, center)

        # write all periods with conditional formatting in weekend schedule
        x = 0
        y = 0
        for month in self.energyweekendschedule:
            for hour in month:
                if hour == 1:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_yellow)
                elif hour == 2:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_blue)
                elif hour == 3:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_green)
                elif hour == 4:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_red)
                elif hour == 5:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_purple)
                elif hour == 6:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_lime)
                else:
                    wksht_weekend.write(1 + y, 1 + x, hour, cond_else)
                x += 1
            x = 0
            y += 1
            # 위 코드 주석과 동일
        # -------------------- rates --------------------
        #  'Rates' 스케줄의 엑셀 파일 워크시트에 기간과 티어를 헤더에 작성하고 해당 열의 너비를 설정하는 부분
        # write period and tiers in header
        header = ['Period', 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5', 'Tier 6', 'Tier 7', 'Tier 8']
        for i in range(len(header)):
            wksht_rates.write(0, i, header[i])
        wksht_rates.set_column(0, 0, 6.14, center)
        wksht_rates.set_column(1, 8, 8.3, center)
        # 'Period', 'Tier 1', 'Tier 2' 등을 헤더로 하는 각 열에 대한 정보를 작성하고, 열의 너비를 설정

        # write period number and subsequent tier rates
        period_number = 1
        count = 0
        for period in self.energy_period_list:
            wksht_rates.write(period_number, 0, period_number)
            for tier in period.tier_list:
                wksht_rates.write(period_number, 1 + count, tier.get_rate())
                count += 1
            count = 0
            period_number += 1
        workbook.close()
        #  'Rates' 스케줄의 엑셀 파일 워크시트에 기간 번호와 해당 기간에 속하는 티어의 요금을 작성하는 부분
        # energy_period_list에서 가져온 값들을 반복하여 해당 값을 엑셀 파일에 작성

    def read_calendar(self):
    # "calendar.xlsx" 파일에서 "Weekday", "Weekend", "Rates" 시트를 읽어와서 각각의 데이터프레임을 출력하는 함수
        """
        After user confirms their excel workbook is complete, each sheet is turned into a data frame

        """
        print(" ")
        file = "calendar.xlsx"
        print("DF_WEEKDAY")
        df_weekday = pd.read_excel(file, sheet_name="Weekday")
        print(df_weekday)
        print(" ")
        print("DF_WEEKEND")
        df_weekend = pd.read_excel(file, sheet_name="Weekend")
        print(df_weekend)
        print(" ")
        print("DF_RATES")
        df_rates = pd.read_excel(file, sheet_name="Rates")
        print(df_rates)
        print(" ")
        # pd.read_excel 함수를 사용하여 엑셀 시트를 데이터프레임으로 변환

    def run(self):
    # 프로그램을 실행하는 메인 함수
        """
        Runs the program utilizing the functions

        """
        self.print_all()
        # 모든 요금제의 식별 정보를 출력
        i = int(input("Which tariff would you like to use?..."))
        # 사용자로부터 요금제 선택을 위한 입력을 받음
        self.print_index(i)
        # 선택한 요금제의 상세 정보를 출력
        self.print_energy_structure()
        # 에너지 구조, 월 및 시간 일정을 출력
        self.calendar()
        # 엑셀 캘린더 파일을 생성
        file = "calendar.xlsx"
        # 생성된 엑셀 파일의 이름을 지정
        # in Windows os, you can edit the spreadsheet here first
        if sys.platform.startswith('win'): # 현재 운영 체제가 Windows인 경우에만 다음 블록을 실행
            os.startfile(file)
            # Windows에서 엑셀 파일을 염
            response = input("Type 'ready' when you are done editing the excel file...")
            # 사용자에게 엑셀 파일 편집이 완료되었음을 입력하도록 요청
            while response != "ready":
                response = input("Type 'ready' when you are done editing the excel file...")
                # 사용자가 "ready"를 입력할 때까지 계속해서 입력을 기다림
        self.read_calendar()
        # 엑셀 캘린더 파일의 내용을 읽어서 출력

def main():
    api = API() # API 클래스의 인스턴스를 생성
    api.run() # 생성된 API 인스턴스를 이용하여 프로그램을 실행

if __name__ == "__main__": main() # 현재 스크립트가 직접 실행되는 경우에만 다음 블록을 실행, main 함수를 호출하여 프로그램을 시작
