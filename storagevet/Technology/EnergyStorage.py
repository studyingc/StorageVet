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
Storage

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER
from storagevet.ErrorHandling import *


class EnergyStorage(DER):
    """ A general template for storage object

    We define "storage" as anything that can affect the quantity of load/power being delivered or used. Specific
    types of storage are subclasses. The storage subclass should be called. The storage class should never
    be called directly.

    """

    def __init__(self, params):
    # 에너지 저장 시스템을 나타내며, 초기화 과정에서 주어진 파라미터들을 사용하여 객체를 설정
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        super().__init__(params)
        # 부모 클래스의 초기화
        # input params
        # note: these should never be changed in simulation (i.e from degradation)
        self.technology_type = 'Energy Storage System'
        # 에너지 저장 시스템의 유형을 나타내는 문자열을 저장
        try:
          self.rte = params['rte']/1e2
        except KeyError:
          self.rte = 1/params['energy_ratio']
        # 'rte' 키가 있으면 params['rte']/1e2 값을, 없으면 1/params['energy_ratio'] 값을 self.rte 속성으로 설정
        self.sdr = params['sdr']/1e2
        self.ene_max_rated = params['ene_max_rated']
        self.dis_max_rated = params['dis_max_rated']
        self.dis_min_rated = params['dis_min_rated']
        self.ch_max_rated = params['ch_max_rated']
        self.ch_min_rated = params['ch_min_rated']
        self.ulsoc = params['ulsoc']/1e2
        self.llsoc = params['llsoc']/1e2
        self.soc_target = params['soc_target']/1e2

        self.fixedOM_perKW = params['fixedOM']  # $/kW
        self.variable_om = params['OMexpenses']*1e-3  # $/MWh * 1e-3 = $/kWh
        self.incl_startup = params['startup']
        self.incl_binary = params['binary']
        self.daily_cycle_limit = params['daily_cycle_limit']
        # self에 해당하는 변수들 라이브러리 값으로 초기화
        self.capital_cost_function = [params['ccost'], params['ccost_kW'], params['ccost_kWh']]

        if self.incl_startup:
            self.p_start_ch = params['p_start_ch']
            self.p_start_dis = params['p_start_dis']

        # to be changed and reset everytime the effective energy capacity changes
        self.effective_soe_min = self.llsoc * self.ene_max_rated
        self.effective_soe_max = self.ulsoc * self.ene_max_rated

        self.variable_names = {'ene', 'dis', 'ch', 'uene', 'uch', 'udis'}
        # 변수의 이름을 나타내는 문자열 세트를 설정
    def discharge_capacity(self):
    # 최대 방전 가능량 반환
        """

        Returns: the maximum discharge that can be attained

        """
        return self.dis_max_rated

    def charge_capacity(self):
    # 최대 충전 가능량 반환
        """

        Returns: the maximum charge that can be attained

        """
        return self.ch_max_rated

    def energy_capacity(self, solution=False):
    # 최대로 가능한 에너지 용량 반환
        """

        Returns: the maximum energy that can be attained

        """
        return self.ene_max_rated

    def operational_max_energy(self):
    # DER에 저장되어야 하는 최대 에너지 반환
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """

        return self.effective_soe_max

    def operational_min_energy(self):
    # DER에 저장되어야 하는 최소 에너지 반환
        """

        Returns: the minimum energy that should stored in this DER based on user inputs
        """

        return self.effective_soe_min

    def qualifying_capacity(self, event_length):
    # RA 또는 DR에 대한 자격을 얻기 위해 DER가 얼마나 많은 전력을 방전할 수 있는지를 설명하며, 이를 통해 시스템의 자격 부여를 결정
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        # RA 또는 DR 이벤트의 길이로, DER가 방전할 것으로 예상되는 총 시간
        return min(self.discharge_capacity(), self.operational_max_energy()/event_length)
        # DER의 방전 용량 및 시스템이 자격 부여를 위해 얼마나 많은 에너지를 방전할 수 있는지를 비교하여 더 작은 값을 반환

    def initialize_variables(self, size):
    # 적화 변수들을 딕셔너리에 추가하는 역할을 합니다. 주어진 크기(size)의 최적화 변수들을 만들고 딕셔너리에 추가
        """ Adds optimization variables to dictionary

        Variables added: (with self.unique_ess_id as a prefix to these)
            ene (Variable): A cvxpy variable for Energy at the end of the time step (kWh)
            dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step (kW)
            ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step (kW)
            on_c (Variable/Parameter): A cvxpy variable/parameter to flag for charging in previous interval (bool)
            on_d (Variable/Parameter): A cvxpy variable/parameter to flag for discharging in previous interval (bool)
            start_c (Variable):  A cvxpy variable to flag to capture which intervals charging started (bool)
            start_d (Variable): A cvxvy variable to flag to capture which intervals discharging started (bool)

        Notes:
            CVX Parameters turn into Variable when the condition to include them is active

        Args:
            size (Int): Length of optimization variables to create

        """
        self.variables_dict = {
            'ene': cvx.Variable(shape=size, name=self.name + '-ene'),
            'dis': cvx.Variable(shape=size, name=self.name + '-dis'),
            'ch': cvx.Variable(shape=size, name=self.name + '-ch'),
            'uene': cvx.Variable(shape=size, name=self.name + '-uene'),
            'udis': cvx.Variable(shape=size, name=self.name + '-udis'),
            'uch': cvx.Variable(shape=size, name=self.name + '-uch'),
            'on_c': cvx.Parameter(shape=size, name=self.name + '-on_c', value=np.ones(size)),
            'on_d': cvx.Parameter(shape=size, name=self.name + '-on_d', value=np.ones(size)),
            'start_c': cvx.Parameter(shape=size, name=self.name + '-start_c', value=np.ones(size)),
            'start_d': cvx.Parameter(shape=size, name=self.name + '-start_d', value=np.ones(size)),
        }
       # 딕셔너리를 초기화하고 최적화 변수들을 추가
       # size는 최적화 변수의 길이

        if self.incl_binary: # incl_binary 속성이 True일 때 이진 변수를 추가하는 코드
            self.variable_names.update(['on_c', 'on_d'])
            self.variables_dict.update({'on_c': cvx.Variable(shape=size, boolean=True, name=self.name + '-on_c'),
                                        'on_d': cvx.Variable(shape=size, boolean=True, name=self.name + '-on_d')})
            # 'on_c'와 'on_d'를 variable_names에 추가
            # 'on_c'와 'on_d'에 대한 이진 변수를 variables_dict에 추가
         
            if self.incl_startup: # incl_startup이 True
                self.variable_names.update(['start_c', 'start_d'])
                self.variables_dict.update({'start_c': cvx.Variable(shape=size, name=self.name + '-start_c'),
                                            'start_d': cvx.Variable(shape=size, name=self.name + '-start_d')})
                # 'start_c'와 'start_d'를 variable_names에 추가
                # 'start_c'와 'start_d'에 대한 변수를 variables_dict에 추가
   
    def get_state_of_energy(self, mask): 
    #  에너지 상태를 반환
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        return self.variables_dict['ene']
        # mask 데이터프레임을 매개변수로 받아와 해당 시점에 에너지 상태를 나타내는 'ene' 변수의 값을 반환

    def get_discharge(self, mask):
    # 해당 시간에 대한 방전 값을 반환
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return self.variables_dict['dis']
        # mask 데이터프레임을 매개변수로 받아와 해당 시점에 방전 값을 나타내는 'dis' 변수의 값을 반환
    
    def get_charge(self, mask):
    # 해당 시간에 대한 충전 값을 반환
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return self.variables_dict['ch']
        # 해당 시점에 충전 값을 나타내는 'ch' 변수의 값을 반환

    def get_capex(self, **kwargs):
    # 주어진 기술의 자본비용(capex)을 반환
        """ Returns the capex of a given technology
        """
        return np.dot(self.capital_cost_function, [1, self.dis_max_rated, self.ene_max_rated])
        # capital_cost_function과 기술의 최대 방전 및 최대 에너지에 대한 가중치
        # np.dot을 사용하여 가중치와 최대 방전, 최대 에너지를 곱한 값을 합산하여 자본비용을 계산

    def get_fixed_om(self):
    # 주어진 기술의 고정 운영 및 유지관리 비용(fixed O&M)을 반환
        """ Returns the fixed om of a given technology
        """
        return self.fixedOM_perKW * self.dis_max_rated
        # 최대 방전량과 단위 최대 방전량당 고정 O&M 비용을 곱하여 총 고정 O&M 비용을 계산

    def get_charge_up_schedule(self, mask):
    # 그리드로 전원을 제공하기 위해 예약할 수 있는 양을 반환
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['ch'] - self.ch_min_rated
        #  'ch' 변수에서 최소 충전 비율을 뺀 값을 반환
        # 충전 스케줄링에 관련된 최적화

    def get_charge_down_schedule(self, mask):
    # 그리드로부터 전력을 가져오기 위해 예약할 수 있는 충전 양을 반환
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.ch_max_rated - self.variables_dict['ch']
        # 최대 충전 비율에서 'ch' 변수 값을 뺀 값을 반환
        # 충전 스케줄링에 관련된 최적화 

    def get_discharge_up_schedule(self, mask):
    # 그리드로 전력을 공급하기 위해 예약할 수 있는 양을 반환
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.dis_max_rated - self.variables_dict['dis']
        # 메서드는 최대 방전 비율에서 'dis' 변수 값을 뺀 값을 반환
        #  방전 스케줄링에 관련된 최적화 

    def get_discharge_down_schedule(self, mask):
    # 그리드로부터 전력을 가져오기 위해 예약할 수 있는 방전 양을 반환
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['dis'] - self.dis_min_rated
        # 'dis' 변수 값에서 최소 방전 비율을 뺀 값을 반환
        # 방전 스케줄링에 관련된 최적화 문제에서 사용

    def get_delta_uenegy(self, mask):
    # 현재 SOE 수준에서 부분 시간대 에너지 이동에 의해 DER의 에너지 수준이 변하는 양을 반환
        """ the amount of energy, from the current SOE level the DER's state of energy changes
        from subtimestep energy shifting

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['uene']
        # 'uene' 변수의 값을 반환
        # 에너지 이동과 관련된 최적화

    def get_uenergy_increase(self, mask):
    # 한 타임 스텝 동안 배포 그리드에 공급되는 에너지 양을 반환
        """ the amount of energy in a timestep that is provided to the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['uch'] * self.dt
        #  'uch' 변수의 값에 시간 간격 dt를 곱한 값을 반환
        # 분산 그리드에 에너지를 제공하는 양과 관련된 최적화

    def get_uenergy_decrease(self, mask):
    # 한 타임 스텝 동안 분배 그리드에서 가져온 에너지 양을 반환
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['udis'] * self.dt
        # 'udis' 변수의 값에 시간 간격 dt를 곱한 값을 반환
        # 분배 그리드에서 에너지를 가져오는 양과 관련된 최적화

    def objective_function(self, mask, annuity_scalar=1):
    # 해당 기술과 관련된 목적 함수를 생성하는 것
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kW
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else annuity_scalar should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """

        # create objective expression for variable om based on discharge activity
        var_om = cvx.sum(self.variables_dict['dis'] + self.variables_dict['udis']) * self.variable_om * self.dt * annuity_scalar
        # 가변 운영 비용 (var_om)은 방전 활동을 기반으로 생성됩
        costs = {
            self.name + ' fixed_om': self.get_fixed_om() * annuity_scalar,
            self.name + ' var_om': var_om
        }
        # add startup objective costs
        if self.incl_startup:
        # 시작 비용이 있는 경우 (incl_startup가 True인 경우) 시작 비용을 목적 함수에 추가
            costs.update({
                self.name + ' ch_startup': cvx.sum(self.variables_dict['start_c']) * self.p_start_ch * annuity_scalar,
                self.name + ' dis_startup': cvx.sum(self.variables_dict['start_d']) * self.p_start_dis * annuity_scalar})
                # fixed_om 및 var_om에 대한 목적 함수 값을 포함하는 딕셔너리 costs를 생성

        return costs
        # 주어진 조건에 따라 고정 운영 및 가변 운영 비용에 대한 목적 함수를 만들어 반환

    def constraints(self, mask, sizing_for_rel=False, find_min_soe=False):
    # 기술의 물리적 제약과 서비스 제약을 나타내는 제약 목록을 생성
    # 기술이 특정 서비스에 대한 제약을 가지지 않는 경우에 사용
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        size = int(np.sum(mask))

        ene_target = self.soc_target * self.effective_soe_max   # this is init_ene

        # optimization variables
        ene = self.variables_dict['ene']
        dis = self.variables_dict['dis']
        ch = self.variables_dict['ch']
        uene = self.variables_dict['uene']
        udis = self.variables_dict['udis']
        uch = self.variables_dict['uch']
        on_c = self.variables_dict['on_c']
        on_d = self.variables_dict['on_d']
        start_c = self.variables_dict['start_c']
        start_d = self.variables_dict['start_d']
        # 얖의 함수들에서 반환된 변수 값들 -> 최적화에 사용
        
        # 최적화 문제에서 에너지 상태 (ene)와 관련된 제약 조건을 정의합니다. 
        # 이러한 제약 조건의 성격은 최적화가 사이징을 위한 것인지 아닌지 (sizing_for_rel)에 따라 다릅
        if sizing_for_rel:
            constraint_list += [
                cvx.Zero(ene[0] - ene_target + (self.dt * dis[0]) - (self.rte * self.dt * ch[0]) - uene[0] + (ene[0] * self.sdr * 0.01))]
            constraint_list += [
                cvx.Zero(ene[1:] - ene[:-1] + (self.dt * dis[1:]) - (self.rte * self.dt * ch[1:]) - uene[1:] + (ene[1:] * self.sdr * 0.01))]
         # 각 시간 단계 (ene[0], ene[1:])에 대해 코드는 해당 시간 단계의 시작에서의 에너지 (ene)가 충전 및 방전을 고려한 경우 목표 에너지 값 (ene_target)과 같아지도록 함
         # 시간 단계 동안의 충전 (ch) 및 방전 (dis)을 고려한 항목뿐만 아니라, 현재 에너지 상태 수준에서의 에너지가 서브-타임스텝 에너지 이동으로 인해 변하는 양을 나타내는 항목 (uene)과 자체 방전 (sdr)과 관련된 항목이 있음
        else:
            # energy at beginning of time step must be the target energy value
            # 각 시간 단계에서 코드는 해당 시간 단계의 시작에서의 에너지 (ene[0])가 목표 에너지 값 (ene_target)과 같아지도록 함
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
            # energy evolution generally for every time step
            # 나머지 시간 단계 (ene[1:])에 대해 코드는 시간 단계 동안의 에너지 변화가 충전, 방전 및 서브-타임스텝 에너지 이동과 일관되도록 합
            # 시간 단계 전체에 걸쳐 에너지 진화를 보장
            constraint_list += [
                cvx.Zero(ene[1:] - ene[:-1] + (self.dt * dis[:-1]) - (self.rte * self.dt * ch[:-1]) - uene[:-1] + (ene[:-1] * self.sdr * 0.01))]

            # energy at the end of the last time step (makes sure that the end of the last time step is ENE_TARGET
            # 최적화 창의 끝에서는 에너지가 마지막 시간 단계에서의 목표 에너지 값과 같아지도록 하는 추가적인 제약 조건
            constraint_list += [cvx.Zero(ene_target - ene[-1] + (self.dt * dis[-1]) - (self.rte * self.dt * ch[-1]) - uene[-1] + (ene[-1] * self.sdr * 0.01))]
       
        #  충전 (ch) 및 방전 (dis) 전력에 대한 제약 조건과 에너지 상태 (ene)에 대한 제약 조건을 정의
        # constraints on the ch/dis power
        # 충전 및 방전 전력 제약
        constraint_list += [cvx.NonPos(ch - (on_c * self.ch_max_rated))] # self.ch_max_rated: 충번 최대 전력 용량
        constraint_list += [cvx.NonPos((on_c * self.ch_min_rated) - ch)] # 최소 전력 용량
        constraint_list += [cvx.NonPos(dis - (on_d * self.dis_max_rated))] # 방전 최대 전력용량
        constraint_list += [cvx.NonPos((on_d * self.dis_min_rated) - dis)]
        # 제약 조건은 충전 및 방전 전력이 정의한 최대 및 최소 전력 용량 내에 있도록 제한

        # constraints on the state of energy
        # 에너지 상태 제약
        constraint_list += [cvx.NonPos(self.effective_soe_min - ene)] # self.effective_soe_min: 기술의 최소 에너지 상태
        constraint_list += [cvx.NonPos(ene - self.effective_soe_max)]
        # 제약 조건은 에너지 상태가 정의한 최소 및 최대 값 내에 있도록 합
        # 최적화 문제에서 에너지 상태를 특정 범위로 제한함으로써 기술의 물리적 특성을 고려

     
        # account for -/+ sub-dt energy -- this is the change in energy that the battery experiences as a result of energy option
        # if sizing for reliability
        # 에너지 옵션 제약 조건
        if sizing_for_rel:
            constraint_list += [cvx.Zero(uene)]
            # sizing_for_rel이 True인 경우, 에너지 옵션은 0으로 제한
        else:
            constraint_list += [cvx.Zero(uene + (self.dt * udis) - (self.dt * uch * self.rte))]
            # 에너지 옵션은 해당 시간 단계에서의 방전 및 충전으로 인한 에너지 변경

        # the constraint below limits energy throughput and total discharge to less than or equal to
        # (number of cycles * energy capacity) per day, for technology warranty purposes
        # this constraint only applies when optimization window is equal to or greater than 24 hours
        # 일일 주기 제한 제약 조건
        if self.daily_cycle_limit and size >= 24: # self.daily_cycle_limit이 True이고 최적화 창이 24시간 이상인 경우
            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                constraint_list += [cvx.NonPos(cvx.sum(dis[day_mask] + udis[day_mask]) * self.dt - self.ene_max_rated * self.daily_cycle_limit)]
                # 각 일(day)에 대해 방전과 에너지 옵션을 합산한 값이 
                # 기술의 에너지 용량과 주어진 일일 주기 제한의 곱 이하임을 나타내는 제약 조건이 추가
        elif self.daily_cycle_limit and size < 24: # 24시간 미만인 경우, 일일 주기 제한은 적용되지 않음
            TellUser.info('Daily cycle limit did not apply as optimization window is less than 24 hours.')

        # note: cannot operate startup without binary
        # 시작 제약 조건
        if self.incl_startup and self.incl_binary: # self.incl_startup 및 self.incl_binary가 모두 True인 경우, 
            # startup variables are positive
            constraint_list += [cvx.NonPos(-start_c)]
            constraint_list += [cvx.NonPos(-start_d)]
            # difference between binary variables determine if started up in
            # previous interval
            constraint_list += [cvx.NonPos(cvx.diff(on_d) - start_d[1:])]
            constraint_list += [cvx.NonPos(cvx.diff(on_c) - start_c[1:])]
            # 기술의 시동(startup) 변수와 해당 시간 단계에서의 충전(on_c) 및 방전(on_d) 변수 간의 관계를 나타내는 제약 조건이 추가
        return constraint_list

    def timeseries_report(self):
    # DER의 최적화 결과를 시계열 데이터프레임으로 요약하는 함수
    # self.variables_df에서 추출한 최적화 변수들을 기존 결과에 추가
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        tech_id = self.unique_tech_id() 
        results = super().timeseries_report()
        solve_dispatch_opt = self.variables_df.get('dis')
        if solve_dispatch_opt is not None:
            results[tech_id + ' Discharge (kW)'] = self.variables_df['dis'] # Discharge: 최적화 결과에서 DER의 방전량
            results[tech_id + ' Charge (kW)'] = -self.variables_df['ch'] # Charge: 최적화 결과에서 DER의 충전량
            results[tech_id + ' Power (kW)'] = \ # Power: 최적화 결과에서 DER의 전력(방전-충전)
                self.variables_df['dis'] - self.variables_df['ch'] 
            results[tech_id + ' State of Energy (kWh)'] = \ # State of Energy: 최적화 결과에서 DER의 에너지 옵션
                self.variables_df['ene']

            results[tech_id + ' Energy Option (kWh)'] =\ # Energy Option: 최적화 결과에서 DER의 에너지 옵션(에너지 상태변화)
                self.variables_df['uene']
            results[tech_id + ' Charge Option (kW)'] = \ # Charge Option: 최적화 결과에서 DER의 충전 옵션(충전의 증가)
                -self.variables_df['uch']
            results[tech_id + ' Discharge Option (kW)'] = \ # Discharge Option: 최적화 결과에서 DER의 방전 옵션(방전의 증가)
                self.variables_df['udis']
            try:
                energy_rating = self.ene_max_rated.value
                # self.ene_max_rated가 CVXPY의 Variable 객체라면 해당 객체의 value 속성을 사용하여 값을 얻음
            except AttributeError:
                energy_rating = self.ene_max_rated
                # AttributeError가 발생하면, self.ene_max_rated를 그대로 energy_rating 변수에 할당

            results[tech_id + ' SOC (%)'] = \ # 최적화 결과에서 DER의 에너지 상태를 에너지 용량에 대한 백분율로 표시
                (self.variables_df['ene'] / energy_rating) * 1e2

        return results
        # 최적화 결과를 사용자 친화적인 형식으로 표시하여 시계열 데이터프레임으로 반환

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
    # 서비스 관련된 데이터프레임을 사용자에게 보고하기 위한 용도로 계산하는 것
    # 해당 보고서는 기술의 배치 맵에 관련된 정보를 담고 있을 것
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        return {f"{self.name.replace(' ', '_')}_dispatch_map": self.dispatch_map()}

    def dispatch_map(self):
    # 에너지 저장 시스템의 순 전력을 가져와서 히트맵(heat map)으로 변환
        """ Takes the Net Power of the Storage System and tranforms it into a heat map

        Returns:

        """
        dispatch = pd.DataFrame(self.variables_df['dis'] - self.variables_df['ch'])
        dispatch.columns = ['Power']
        # 에너지 저장 시스템의 순 전력을 가져와서 히트맵(heat map)으로 변환
        # 새로운 데이터프레임인 dispatch에 저장하고, 열 이름을 'Power'로 변경
        dispatch.loc[:, 'date'] = self.variables_df.index.date
        dispatch.loc[:, 'hour'] = (self.variables_df.index + pd.Timedelta('1s')).hour + 1
        dispatch = dispatch.reset_index(drop=True)
        dispatch_map = dispatch.pivot_table(values='Power', index='hour', columns='date')
        # 날짜 및 시간 정보를 dispatch 데이터프레임에 추가
        # 날짜 및 시간을 기준으로 데이터를 재배열하여 dispatch_map 데이터프레임을 생성
        return dispatch_map

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):\
    # 해당 기술에 대한 프로포마(견적서)를 생성하는 것
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        # 상위 클래스에서 정의된 공통의 프로포마 항목을 가져옴
        if self.variables_df.index.empty:
            # 만약 self.variables_df가 비어 있다면, pro_forma를 반환
            return pro_forma
        optimization_years = self.variables_df.index.year.unique()
        # 인덱스에서 유일한 연도를 가져와 최적화 연도를 결정
        tech_id = self.unique_tech_id()
        startup_costs = pd.DataFrame()
        # OM COSTS
        om_costs = pd.DataFrame()
        cumulative_energy_dispatch_kw = pd.DataFrame()
        dis = self.variables_df['dis']
        udis = self.variables_df['udis']
        dis_column_name = tech_id + ' Cumulative Energy Dispatch (kW)'
        variable_column_name = tech_id + ' Variable O&M Cost'
        # 변수 및 데이터프레임 초기화
       
        for year in optimization_years:
        #  최적화 연도별로 여러 항목을 계산하고 결과 데이터프레임에 추가하는 부분
            index_yr = pd.Period(year=year, freq='y')
            # add fixed o&m costs
            om_costs.loc[index_yr, self.fixed_column_name()] = -self.fixedOM_perKW
            # add variable o&m costs
            dis_sub = dis.loc[dis.index.year == year]
            udis_sub = udis.loc[udis.index.year == year]
            om_costs.loc[index_yr, variable_column_name] = -self.variable_om
            # om_costs 데이터프레임에 고정 O&M 비용 및 가변 O&M 비용을 추가
            cumulative_energy_dispatch_kw.loc[index_yr, dis_column_name] = np.sum(dis_sub) + np.sum(udis_sub)
            # cumulative_energy_dispatch_kw 데이터프레임에 누적 에너지 방출 값을 추가

            # add startup costs
            if self.incl_startup:
                start_c_sub = self.variables_df['start_c'].loc[self.variables_df['start_c'].index.year == year]
                startup_costs.loc[index_yr, tech_id + ' Start Charging Costs'] = -np.sum(start_c_sub * self.p_start_ch)
                start_d_sub = self.variables_df['start_d'].loc[self.variables_df['start_d'].index.year == year]
                startup_costs.loc[index_yr, tech_id + ' Start Discharging Costs'] = -np.sum(start_d_sub * self.p_start_dis)
                # start_c_sub 및 start_d_sub를 사용하여 각각 충전 및 방전을 시작하는 데 필요한 이벤트의 시작 비용을 가져옴
                # startup_costs 데이터프레임에 해당 비용을 추가
                # "Start Charging Costs" 및 "Start Discharging Costs" 열에 누적된 시작 비용이 기록

        # fill forward (escalate rates)
        if self.incl_startup:
            startup_costs = fill_forward_func(startup_costs, None)
        om_costs = fill_forward_func(om_costs, None, is_om_cost = True)
        # 기동 비용 및 O&M 비용에 대한 가격 인상율 적용을 위해 fill_forward_func 함수를 사용
        # 비용을 에스컬레이션하고 나서 프로포마 보고서에 추가

        # interpolate cumulative energy dispatch between analysis years
        #   be careful to not include years labeled as Strings (CAPEX)
        years_list = list(filter(lambda x: not(type(x) is str), om_costs.index))
        analysis_start_year = min(years_list).year
        analysis_end_year = max(years_list).year
        cumulative_energy_dispatch_kw = self.interpolate_energy_dispatch(
            cumulative_energy_dispatch_kw, analysis_start_year, analysis_end_year, None)
        # 누적 에너지 송출을 분석 년도 사이에 보간
        # cumulative_energy_dispatch_kw DataFrame이 분석 시작 및 종료 연도에 대해 채워지도록 보장
        # calculate om costs in dollars, as rate * energy
        # fixed om
        om_costs.loc[:, self.fixed_column_name()] = om_costs.loc[:, self.fixed_column_name()] * self.dis_max_rated
        # 고정 O&M 비용: 고정 O&M 비용은 장치의 최대 방전 용량과 관련하여 확장
        # variable om
        om_costs.loc[:, variable_column_name] = om_costs.loc[:, variable_column_name] * self.dt * cumulative_energy_dispatch_kw.loc[:, dis_column_name]
        # 가변 O&M 비용: 가변 O&M 비용은 각 년도에 발생한 누적 에너지 송출 양에 대한 비용

        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        if self.incl_startup:
            pro_forma = pd.concat([pro_forma, startup_costs], axis=1)
        # pro_forma에 추가: 계산된 비용은 프로포마와 병합되어 최종 보고서를 생성

        return pro_forma

    def verbose_results(self):
    # 해당 value stream에 대한 추가 정보를 수집하여 반환
        """ Results to be collected iff verbose -- added to the opt_results df

        Returns: a DataFrame

        """
        results = pd.DataFrame(index=self.variables_df.index)
        return results
