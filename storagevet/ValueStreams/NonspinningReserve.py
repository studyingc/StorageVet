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
"""
NonspinningReserve.py

이 Python 클래스에는 StorageVet 내의 서비스 분석에 특정한 메서드와 속성이 포함되어 있습니다
"""
from storagevet.ValueStreams.MarketServiceUp import MarketServiceUp
import cvxpy as cvx
import storagevet.Library as Lib


class NonspinningReserve(MarketServiceUp):
    """  비회전 예비전력. 각 서비스는 ValueStream 클래스의 자식 클래스로 될 것입니다.
     이 클래스는 비회전 예비전력 시장 서비스를 나타냅니다. MarketServiceUp 클래스를 상속받습니다.
    """

    def __init__(self, params):
        """ 목적 함수를 생성하고 제약 조건을 찾아 생성합니다.

        Args:
            params (Dict): 입력 매개변수
        """
     # 부모 클래스(MarketServiceUp)의 생성자를 호출합니다.
        super(NonspinningReserve, self).__init__('NSR', 'Non-spinning Reserve', params)
     # 시계열 제약 조건이 활성화되어 있는지 확인합니다.
        self.ts_constraints = params.get('ts_constraints', False)
     # 시계열 제약 조건이 활성화되어 있다면, 추가 속성을 설정합니다.
        if self.ts_constraints:
            self.max = params['max']
            self.min = params['min']

    def grow_drop_data(self, years, frequency, load_growth):
        """ 주어진 데이터를 성장시키거나 추가로 들어온 데이터를 제거하여 데이터를 갱신합니다.
        최적화 실행 전에 add_growth_data 이후에 이 메서드를 호출해야 합니다.
        Args:
            years (List): 분석이 수행될 연도 목록
            frequency (str): 시계열 데이터의 주기
            load_growth (float): 이 시뮬레이션의 부하 성장률의 백분율/소수값


        """
     # 부모 클래스의 grow_drop_data 메서드 호출
        super().grow_drop_data(years, frequency, load_growth)
     # 시계열 제약 조건이 활성화된 경우 추가 데이터를 채우고 불필요한 데이터를 제거합니다.
        if self.ts_constraints:
         # 최댓값에 대한 추가 데이터 채우기 및 불필요한 데이터 제거
            self.max = Lib.fill_extra_data(self.max, years, self.growth, frequency)
            self.max = Lib.drop_extra_data(self.max, years)
         # 최솟값에 대한 추가 데이터 채우기 및 불필요한 데이터 제거
            self.min = Lib.fill_extra_data(self.min, years, self.growth, frequency)
            self.min = Lib.drop_extra_data(self.min, years)

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating):
        """기본 제약 조건 목록을 생성하는 메서드입니다. 제약 조건이 없는 서비스에서 사용됩니다.
        Args:
            mask (DataFrame): subs 데이터 집합에 포함된 time_series 데이터에 해당하는 인덱스에 대한 참인 부울 배열
            tot_variable_gen (Expression): variable/intermittent generation sources의 
            load_sum (list, Expression): 시스템 내의 부하 합계
            generator_out_sum (list, Expression): 시스템 내에서의 발전 합계
            net_ess_power (list, Expression): 시스템 내의 모든 ESS의 순 전력의 합계. grid의 흐름은 음수
            combined_rating (Dictionary): 각 DER 클래스 유형의 결합 등급

        Returns:
            나중에 제약 조건을 집계하기 위한 빈 목록
        """
     # 부모 클래스의 constraints 메서드 호출
        constraint_list = super().constraints(mask, load_sum, tot_variable_gen,
                                              generator_out_sum, net_ess_power, combined_rating)
       # 시계열 제약 조건이 활성화된 경우 서비스 참여 제약 조건 추가
       # Max와 Min은 ch_less 및 dis_more의 합을 제한합니다.
        if self.ts_constraints:
            constraint_list += \
                [cvx.NonPos(self.variables['ch_less'] +
                            self.variables['dis_more'] - self.max.loc[mask])]
            constraint_list += \
                [cvx.NonPos(-self.variables['ch_less'] -
                            self.variables['dis_more'] + self.min.loc[mask])]

        return constraint_list

    def timeseries_report(self):
        """ 이 Value Stream에 대한 최적화 결과를 요약합니다.

        Returns:  이 인스턴스와 관련된 결과를 요약하는 사용자 친화적인 열 헤더를 가진 시계열 데이터프레임

        """
     # 부모 클래스의 timeseries_report 메서드 호출
        report = super().timeseries_report()
     # 시계열 제약 조건이 활성화된 경우 max 및 min 열을 결과에 추가
        if self.ts_constraints:
            report.loc[:, self.max.name] = self.max
            report.loc[:, self.min.name] = self.min
        return report

    def min_regulation_down(self):
      """최소 규제 감소량을 반환합니다.

    Returns:
        시계열 제약 조건이 활성화된 경우 self.min을 반환하고, 그렇지 않으면 부모 클래스의 min_regulation_down 메서드의 반환 값을 사용합니다.
    """
        if self.ts_constraints:
            return self.min
        return super().min_regulation_down()

    def max_participation_is_defined(self):
        return hasattr(self, 'max')
