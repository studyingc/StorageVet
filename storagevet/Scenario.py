"""Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import time
import copy
from datetime import date
import calendar
from storagevet.ValueStreams.DAEnergyTimeShift import DAEnergyTimeShift
from storagevet.ValueStreams.FrequencyRegulation import FrequencyRegulation
from storagevet.ValueStreams.NonspinningReserve import NonspinningReserve
from storagevet.ValueStreams.DemandChargeReduction import DemandChargeReduction
from storagevet.ValueStreams.EnergyTimeShift import EnergyTimeShift
from storagevet.ValueStreams.SpinningReserve import SpinningReserve
from storagevet.ValueStreams.Backup import Backup
from storagevet.ValueStreams.Deferral import Deferral
from storagevet.ValueStreams.DemandResponse import DemandResponse
from storagevet.ValueStreams.ResourceAdequacy import ResourceAdequacy
from storagevet.ValueStreams.UserConstraints import UserConstraints
from storagevet.ValueStreams.VoltVar import VoltVar
from storagevet.ValueStreams.LoadFollowing import LoadFollowing
from storagevet.Technology.BatteryTech import Battery
from storagevet.Technology.CAESTech import CAES
from storagevet.Technology.PVSystem import PV
from storagevet.Technology.InternalCombustionEngine import ICE
from storagevet.Technology.Load import Load
from storagevet.ServiceAggregator import ServiceAggregator
from storagevet.POI import POI
import storagevet.Finances as Fin
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class Scenario(object):
    """ A scenario is one simulation run in the model_parameters file.

    """
    TECH_CLASS_MAP = {
        'CAES': CAES, # 압축 공기 에너지 저장
        'Battery': Battery, # 배터리
        'PV': PV, # 태양 전지
        'ICE': ICE, # 내연 기관
        'Load': Load # 부하
    } # 기술을 해당하는 클래스에 매핑하여 프로그램에서 쉽게 접근할 수 있도록 함
    VS_CLASS_MAP = {
        'Deferral': Deferral, # 지연 서비스
        'DR': DemandResponse, # 수요 반응
        'RA': ResourceAdequacy, # 자원 보증
        'Backup': Backup, # 백업 서비스
        'Volt': VoltVar,  # THIS DOES NOT WORK
        'User': UserConstraints, # 사용자 제약 조건
        'DA': DAEnergyTimeShift, # 일간 에너지 시간 이동
        'FR': FrequencyRegulation, # 주파수 조절
        'LF': LoadFollowing, # 부하 추적
        'SR': SpinningReserve, # 회전예비력
        'NSR': NonspinningReserve, # 비회적 예비력
        'DCM': DemandChargeReduction, # 수요 요금 감소 
        'retailTimeShift': EnergyTimeShift, # 소매 시간 이동
    } # 가상 서비스를 해당하는 클래스에 매핑하여 프로그램에서 쉽게 접근할 수 있도록 함

    def __init__(self, input_tree):
        """ 시나리오 객체를 초기화하는 함수
        사용자 입력 파라미터 및 초기화된 속성을 설정함함
        """
        self.verbose = input_tree.Scenario['verbose']
        # verbose 속성은 input_tree의 Scenario 딕셔너리에서 'verbose' 키에 해당하는 값을 가져와 할당
        self.start_time = time.time()
        self.start_time_frmt = time.strftime('%Y%m%d%H%M%S')
        self.end_time = 0

        # add general case params (USER INPUTS)
        self.dt = input_tree.Scenario['dt']
        self.verbose_opt = input_tree.Scenario['verbose_opt']
        self.n = input_tree.Scenario['n']
        # self.n_control = input_tree.Scenario['n_control']
        self.n_control = 0
        self.mpc = input_tree.Scenario['mpc']

        self.start_year = input_tree.Scenario['start_year']
        self.end_year = input_tree.Scenario['end_year']
        self.opt_years = input_tree.Scenario['opt_years']
        self.incl_binary = input_tree.Scenario['binary']
        self.incl_slack = input_tree.Scenario['slack']
        self.def_growth = input_tree.Scenario['def_growth']/100
        self.frequency = input_tree.Scenario['frequency']

        # save inputs to be used to initialize classes later
        self.technology_inputs_map = {
            'CAES': input_tree.CAES,
            'Battery': input_tree.Battery,
            'PV': input_tree.PV,
            'ICE': input_tree.ICE,
            'Load': input_tree.Load
        }
        self.value_stream_input_map = {
            'Deferral': input_tree.Deferral,
            'DR': input_tree.DR,
            'RA': input_tree.RA,
            'Backup': input_tree.Backup,
            'Volt': input_tree.Volt,
            'User': input_tree.User,
            'DA': input_tree.DA,
            'FR': input_tree.FR,
            'LF': input_tree.LF,
            'SR': input_tree.SR,
            'NSR': input_tree.NSR,
            'DCM': input_tree.DCM,
            'retailTimeShift': input_tree.retailTimeShift,
            # input_tree 객체의 속성들을 특정 문자열과 연결하여 딕셔너리에 저장하는 역할
        }
        self.poi_inputs = input_tree.POI
        self.finance_inputs = input_tree.Finance

        # these are attributes that are changed as the scenario is solved
        self.solvers = []
        self.poi = None # 관심 지점
        self.service_agg = None # 서비스 집계
        self.cost_benefit_analysis = None #비용 편익 분석
        self.optimization_levels = pd.DataFrame() # 최적화 수준
        self.objective_values = pd.DataFrame() # 목적 함수 값
        self.system_requirements = None # 시스템 요구 사항
        self.opt_engine = True  # indicates that dervet should go to the optimization module and size there
        # 의심해볼만한 점

    def set_up_poi_and_service_aggregator(self, point_of_interconnection_class=POI, service_aggregator_class=ServiceAggregator):
        """ POI(접속 지점)와 서비스 어그리게이터를 초기화하는 함수
        """
        # these need to be initialized after opt_agg is created
        self.poi = point_of_interconnection_class(self.poi_inputs, self.technology_inputs_map, self.TECH_CLASS_MAP)
        self.service_agg = service_aggregator_class(self.value_stream_input_map, self.VS_CLASS_MAP)
        if self.service_agg.is_deferral_only():
            TellUser.warning("Only active Value Stream is Deferral, so not optimizations will be solved...")
            self.opt_engine = False

    def initialize_cba(self):
        """ StorageVET의 제한된 비용-편익 분석 모듈을 초기화하는 함수
        """
        self.cost_benefit_analysis = Fin.Financial(self.finance_inputs, self.start_year, self.end_year)
        # add fuel_cost to active DERs that can consume fuel
        for der in self.poi.der_list:
            der.set_fuel_cost(self.cost_benefit_analysis.get_fuel_cost)

    def fill_and_drop_extra_data(self):
        """ 값 스트림 및 기술 요소에서 분석 연도에 필요한 데이터를 유지하고 추가 데이터를 필요에 따라 추가하는 함수수
        """

        # self는 객체의 인스턴스 그 자체를 말한다. 즉, 객체 자기 자신을 참조하는 매개변수
        
        # update_analysis_years 메서드를 사용하여 분석 연도를 업데이트하고 필요한 데이터를 유지(연도 간의 변화 및 시스템의 노출 검사)
        self.opt_years = self.service_agg.update_analysis_years(self.end_year, self.poi, self.frequency, self.opt_years, self.def_growth) # class ServiceAggregator 
        
        # 모든 활성 ess의 rte_list를 각 value stream에 추가
        for service in self.service_agg.value_streams.values(): # class ServiceAggregator 
            service.rte_list(self.poi) # class Valuestream

        # 각 값 스트림에 누락된 연도의 데이터를 추가
        for service in self.service_agg.value_streams.values():
            service.grow_drop_data(self.opt_years, self.frequency, self.def_growth)

        # 분석에 사용하지 않을 데이터를 제거
        for der in self.poi.der_list:
            der.grow_drop_data(self.opt_years, self.frequency, self.def_growth) # DER.py 

        # create optimization levels
        self.optimization_levels = self.assign_optimization_level(self.opt_years, self.n, 0, self.frequency, self.dt) # class valuestream

        # ESS 객체 내 degredation 모듈 초기화한다. 만약 해당 ESS 기술에 대해 어떠한 degradation module도 정의되어 있지 않다면, 메서드가 호출되더라도 아무런 동작이나 변경이 발생하지 않을 것
        for der in self.poi.der_list:
            if der.technology_type == "Energy Storage System":
                # optimization_levels를 사용하여 degradation 모듈을 초기화
                der.initialize_degradation_module(self.optimization_levels) # init - module => Batterytech  

        # 시스템 요구 사항 계산 및 Value stream 에 의해 설정된 값이 충족되는지 확인
        self.system_requirements = self.service_agg.identify_system_requirements(self.poi.der_list, self.opt_years, self.frequency) # class ServiceAggregator 

""" 클래스 내부 속성들을 업데이트하고 상태를 변경하는 목적으로 사용되어 반환하는 값이 없는 경우가 많음.
"""
    @staticmethod
     def assign_optimization_level(analysis_years, control_horizon, predictive_horizon, frequency, dt):
        """ 최적화 수준을 할당하는 함수수
        """
        # create dataframe to fill
        level_index = Lib.create_timeseries_index(analysis_years, frequency) # 내부 프로그램에서 시간 단위 시작을 나타냄
        # Lib.create_timeseries_index 함수 활용 -> analysis_years, frequency 기반 인덱 
        level_df = pd.DataFrame({'control': np.zeros(len(level_index))}, index=level_index)
        # level_index를 control이라는 열로 초기
        current_control_level = 0 # 현재 제어 레벨을 0으로 초기화
        # control level should not overlap multiple years & there is only one per timestep
        for yr in level_index.year.unique(): # leverl_index의 고유한 연도 값에 대해서 반복 실
            sub = copy.deepcopy(level_df[level_df.index.year == yr]) 
            # copy.deepcopy 함수를 이용해 해당 연도에 해당하는 dataframe 복사
            if control_horizon == 'year': # control_horizon 값이 year인 경우
                
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level + 1
                # 이전 연도의 최적화 문제(opt_agg)에서부터 계속해서 숫자를 셈
            elif control_horizon == 'month': # control_horizon 값이 month인 경우
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level + sub.index.month
            else:
                # n is number of hours
                control_horizon = int(control_horizon)
                sub['ind'] = range(len(sub))
                # split year into groups of n days
                ind = (sub.ind // (control_horizon / dt)).astype(int) + 1
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = ind + current_control_level
            current_control_level = max(level_df.control)
            # 이전에 설정된 최대 제어 수준을 기반으로 다음 연도의 최적화 문제에 대한 수준을 설정

        # predictive level can overlap multiple years & there can be 1+ per timestep
        if not predictive_horizon: # 만약 perdictive_horizon이 없다
            # set to be the control horizon
            level_df['predictive'] = level_df.loc[:, 'control']
            # 예측 level을 설정하고 이를 control level과 동일하게 설정
        else:
            # TODO this has not been tested yet -- HN (sorry hmu and I will help)
            # create a list of lists
            max_index = len(level_df['control'])
            predictive_level = np.repeat([], max_index)
            current_predictive_level_beginning = 0 # 시작을 나타내는 변수 초기화
            current_predictive_level = 0 # 변수 초기화

            for control_level in level_df.control.unique(): # level_df에서 control_level에 대해 반
                if predictive_horizon == 'year': # predictive_horizon이 year인 경우
                    # needs to be a year from the beginning of the current predictive level, determine
                    # length of the year based on first index in subset
                    start_year = level_index[current_predictive_level_beginning].year[0]
                    # current_predictive_level_beginning에서 시작해 해당 연도의 start_year을 결정
                    f_date = date(start_year, 1, 1) # 시작 연도의 1월 1일을 나타냄
                    l_date = date(start_year + 1, 1, 1) # 다염 연도의 1월 1일을 나타
                    delta = l_date - f_date # 연도의 길이를 구함
                    current_predictive_level_end = int(delta.days*dt) # 연도의 길이를 시간 간격에 맞게 조정해 current_perdictive_level_end를 결정정

                elif predictive_horizon == 'month': # predictive_horizon이 month인 경우
                    # needs to be a month from the beginning of the current predictive level, determine
                    # length of the month based on first index in subset
                    start_index = level_index[current_predictive_level_beginning] # 해당 월의 인덱스 결정 
                    current_predictive_level_end = calendar.monthrange(start_index.year, start_index.month) 
                    # 해당 월의 첫 번째 날과 마지막 날을 사용해 예측 수준의 길이를 계싼
                    current_predictive_level_end = calendar.monthrange(start_index.year, start_index.month)
                else:
                    current_predictive_level_end = predictive_horizon * dt
                # make sure that CURRENT_PREDICTIVE_LEVEL_END stays less than or equal to MAX_INDEX
                current_predictive_level_end = min(current_predictive_level_end, max_index)
                # add CURRENT_PREDICTIVE_LEVEL to lists between CURRENT_PREDICTIVE_LEVEL_BEGINNING and CURRENT_PREDICTIVE_LEVEL_END
                update_levels = predictive_level[current_predictive_level_beginning, current_predictive_level_end]
                update_levels = [dt_level.append(current_predictive_level) for dt_level in update_levels]
                predictive_level[current_predictive_level_beginning, current_predictive_level_end] = update_levels
                current_predictive_level_beginning = np.sum(level_df.control == control_level)
                # increase CURRENT_PREDICTIVE_LEVEL
                current_predictive_level += 1
            level_df['predictive'] = predictive_level
        return level_df

    def optimize_problem_loop(self):
        """ 최적화 루프를 시작하여 다양한 최적화 윈도우에 대한 최적화 문제를 해결하는 함수
        """
        # 만약 최적화 엔진이 없다면 함수 종료
        if not self.opt_engine:
            return
        # 로그 레벨을 info로 설정하여 "Starting optimization loop" 메시지를 출력
        TellUser.info("Starting optimization loop")
        # optimization_level에서 중복되지 않는 예측 기간 값을 가져와 루프를 실행
        for opt_period in self.optimization_levels.predictive.unique():
            # setup + run optimization then return optimal objective costs
            # 최적화를 위한 함수, 제약조건 및 서브 인덱스 설정
            functions, constraints, sub_index = self.set_up_optimization(opt_period)

            ##NOTE: 주석 처리된 부분은 디버깅을 위한 print 문으로, 최종 제약조건과 비용을 출력
            #print(f'\nFinal constraints ({len(constraints)}):')                   # 최종 제약 조건
            #print(f'\nconstraints ({len(constraints)}):')                         # 제약조건
            #print('\n'.join([f'{i}: {c}' for i, c in enumerate(constraints)]))    # 
            #print(f'\ncosts ({len(functions)}):')                                 # 비용
            #print('\n'.join([f'{k}: {v}' for k, v in functions.items()]))         # 
            #print()
            
            # 최적화 문제를 해결하고 최적화 문제, 목적 함수 표현식 및 CVXPY 라이브러리에서 발생한 오류 메시지를 반환
            cvx_problem, obj_expressions, cvx_error_msg = self.solve_optimization(functions, constraints)
            # 최적화 결과를 저장하는 메서드를 호출하여 결과를 저장
            self.save_optimization_results(opt_period, sub_index, cvx_problem, obj_expressions, cvx_error_msg)

    def set_up_optimization(self, opt_window_num, annuity_scalar=1, ignore_der_costs=False):
        """
        연도의 일부를 대상으로 최적화를 설정하고 실행합니다
        """

        # mask: 조건에 부합하는 데이터를 골라낼때 사용 = 필터링
        
        # used to select rows from time_series relevant to this optimization window
        mask = self.optimization_levels.predictive == opt_window_num 
        # 최적화 창(opt_window_num과 같은 값을 가지는 predictive 열)에 관련된 행을 선택
        sub_index = self.optimization_levels.loc[mask].index
        # 최적화 창에 해당하는 인덱스
        TellUser.info(f"{time.strftime('%H:%M:%S')} Running Optimization Problem starting at {sub_index[0]} hb")
        # 최적화 문제 시작하는 시간과, 최적화 창의 첫 번째 인덱스 출력
        opt_var_size = int(np.sum(mask))
        # mask의 true 값의 합을 계산하여 최적화 변수의 크기를 나타냄

        # 변수 설정
        self.poi.initialize_optimization_variables(opt_var_size) #init - variables => POI
        self.service_agg.initialize_optimization_variables(opt_var_size) #init - variables => service_agg

        # POI에서 목적 함수 및 제약 조건 계산에 필요한 값 가져오기
        load_sum, var_gen_sum, gen_sum, tot_net_ess, der_dispatch_net_power, total_soe, agg_p_in, agg_p_out, agg_steam, agg_hotwater, agg_cold = self.poi.get_state_of_system(mask) # get - system => POI
        # 최적화 변수 가져오기
        combined_rating = self.poi.combined_discharge_rating_for_reliability() # combined - reliability => POI
        # ESS와 ICE(발전기)의 결합 방전 등급
        # 여러 개의 전력장치나 전력설비가 함께 연결되어 작동할 때 견딜 수 있는 총 전력/부하 = 전체 시스템이 안정적으로 운영될 수 있는 최대 전력

        # set up controller first to collect and provide inputs to the POI
        funcs, consts = self.service_agg.optimization_problem(mask, load_sum, var_gen_sum, gen_sum, tot_net_ess, combined_rating, annuity_scalar)

        # add optimization problem portion from the POI
        temp_objectives, temp_consts = self.poi.optimization_problem(mask, agg_p_in, agg_p_out, agg_steam, agg_hotwater, agg_cold, annuity_scalar)
        if not ignore_der_costs:
            #  don't ignore der costs
            funcs.update(temp_objectives)
        consts += temp_consts

        # 시스템 요구 사항 제약 조건 추가 (현재 최적화 창에 적용되는 데이터 하위 집합을 가져옴)
        for req_name, requirement in self.system_requirements.items(): # self.system_requirements의 req_name, requirement에 대해서 반복
    
            # NOTE: der_dispatch_net_power is (charge - discharge) for each DER that can dispatch power
            #           (not Intermittent Resources, and not Load) : 간헐 발전원 / 부하가 아님
           
            if req_name == 'der dispatch discharge min': # 요구 사항 이름 'der dispatch discharge min'인 경우
                req_value = requirement.get_subset(mask) # 하위 집합을 가져옴
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')  
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='DerDispatchDisMinReq') 
                consts += [cvx.NonPos(req_parameter + der_dispatch_net_power)] 
                # der_dispatch_net_power의 합이 음수가 되는 제약
                continue

            #if req_name == 'der dispatch charge max':
            #    req_value = requirement.get_subset(mask)
            #    #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
            #    req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='DerDispatchChMaxReq')
            #    consts += [cvx.NonPos(der_dispatch_net_power + -1 * req_parameter)]
            #    continue

            if req_name == 'poi export min': # poi export min인 경우
                req_value = requirement.get_subset(mask) 
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiExportMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * agg_p_out)]
                # 수출 값이 양수일 때 사용됨
                continue

            if req_name == 'poi export max': # poi export max인 경우
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiExportMaxReq')
                consts += [cvx.NonPos(agg_p_out + -1 * req_parameter)]
                # 수출 값이 음수일 때 사용
                continue

            if req_name == 'poi import min': 
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiImportMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * agg_p_in)]
                # 수입 값이 양수일 때 사용
                continue

            if req_name == 'poi import max':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiImportMaxReq')
                consts += [cvx.NonPos(agg_p_in + -1 * req_parameter)]
                # 수입 값이 음수일 때 사용
                continue

            if req_name == 'energy min':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='SysEneMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * total_soe)]
                # 에너지 저장소의 순 에너지가 양수일 때 사용
                continue

            if req_name == 'energy max':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='SysEneMaxReq')
                consts += [cvx.NonPos(total_soe + -1 * req_parameter)]
                # 에너지 저장소의 순 에너지가 음수일 때 사용
                continue

            # if this part of the method is reached, we have failed to recognize a system requirement and should fail
            error_message = f'This system requirement: "{req_name}" is not properly specified by the Scenario class. Cannot continue.'
            TellUser.error(error_message)
            raise SystemRequirementsError(error_message)

        res_dis_d, res_dis_u, res_ch_d, res_ch_u, ue_prov, ue_stor, worst_ue_pro, worst_ue_sto = self.service_agg.aggregate_reservations(mask) # aggregate_reservations => servic aggregator
        sch_dis_d, sch_dis_u, sch_ch_d, sch_ch_u, ue_decr, ue_incr, total_dusoe = self.poi.aggregate_p_schedules(mask) # aggregate_p_schedules => POI
        
        
        # make sure P schedule matches the P reservations
        consts += [cvx.NonPos(res_dis_d + (-1) * sch_dis_d)] # 방전 스케줄이 예약된 용량을 초과하지 않도록 보장
        consts += [cvx.NonPos(res_dis_u + (-1) * sch_dis_u)] # 역방전 스케줄이 예약된 용량을 초과하지 않도록 보장
        consts += [cvx.NonPos(res_ch_u + (-1) * sch_ch_u)] # 역충전 스케줄이 예약된 용량을 초과하지 않도록 보장
        consts += [cvx.NonPos(res_ch_d + (-1) * sch_ch_d)] # 충전 스케줄이 예약된 용량을 초과하지 않도록 보장

        # match uE delta to uE reservation: energy increase
        consts += [cvx.Zero(ue_prov + (-1) * ue_decr)] # 에너지 공급 증가(ue_prov)와 에너지 감소 예약(ue_decr)의 차이가 0이 되도록 제약을 추가
        # match uE delta to uE reservation: energy decrease
        consts += [cvx.Zero(ue_stor + (-1) * ue_incr)] # 에너지 저장 증가(ue_stor)와 에너지 증가 예약(ue_incr)의 차이가 0이 되도록 제약을 추가
        # make sure that the net change in energy is less than the total change in system SOE
        consts += [cvx.NonPos(total_dusoe + (-1) * ue_prov + (-1) * ue_stor)] # 시스템의 순 에너지 변화(total_dusoe)가 에너지 공급 및 저장의 증가(ue_prov, ue_stor)와 음수가 되도록 제약을 추가

        # require that SOE +/- worst case stays within bounds of DER mix
        _, _, soe_limits = self.poi.calculate_system_size() # Caluculate - size => POI
        consts += [cvx.NonPos(total_soe + worst_ue_sto - soe_limits[0])] # 시스템의 총 SOE(total_soe)와 최악의 에너지 저장(worst_ue_sto)가 SOE 한계(soe_limits[0])를 초과하지 않도록 제약을 추가 
        consts += [cvx.NonPos(soe_limits[1] + worst_ue_pro + (-1)*total_soe)] # SOE 한계 상한(soe_limits[1])과 최악의 에너지 공급(worst_ue_pro)이 총 SOE를 초과하지 않도록 제약을 추가

        return funcs, consts, sub_index
        # 함수(funcs), 상수(consts), 및 부속 인덱스(sub_index)가 반환
        # 최적화 문제에 대한 제약 조건을 설정하는 함수에서 이들을 반환하여 최적화 문제를 해결하는 데 필요한 모든 요소를 제공
    
    def solve_optimization(self, obj_expression, obj_const, force_glpk_mi=False):
        """ 최적화 문제를 설정하고 해결하는 함수
        """
        # summary of objective expressions to set up optimization problem
        obj = cvx.Minimize(sum(obj_expression.values()))
        prob = cvx.Problem(obj, obj_const)
        TellUser.info("Finished setting up the problem. Solving now.")
        cvx_error_msg = ''
        try:
            if prob.is_mixed_integer():
                # MBL: GLPK will solver to a default tolerance but can take a long time. Can use ECOS_BB which uses a branch and bound method
                # and input a tolerance but I have found that this is a more sub-optimal solution. Would like to test with Gurobi
                # information on solvers and parameters here: https://www.cvxpy.org/tgitstatutorial/advanced/index.html

                # prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB, mi_abs_eps=1, mi_rel_eps=1e-2, mi_max_iters=1000)
                start = time.time()
                TellUser.debug("glpk_mi solver")
                prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                end = time.time()
                TellUser.info("Time it takes for solver to finish: " + str(end - start))
            else:
                start = time.time()
                # ECOS is default solver and seems to work fine here, however...
                # a problem with ECOS was found when running projects with thermal loads,
                # so we force use of glpk_mi for these cases
                if force_glpk_mi:
                    TellUser.debug("glpk_mi solver (for cases with thermal loads)")
                    prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                else:
                    TellUser.debug("ecos_bb solver")
                    prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB)
                end = time.time()
                TellUser.info("Time it takes for solver to finish: " + str(end - start))
        except (cvx.error.SolverError, RuntimeError) as e:
            TellUser.error("The solver was unable to find a solution.")
            cvx_error_msg = e
        return prob, obj_expression, cvx_error_msg

    def save_optimization_results(self, opt_window_num, sub_index, prob, obj_expression, cvx_error_msg):
        """ 최적화 결과를 저장하고 문제의 해결 상태를 확인하는 함
        opt_window_num (최적화 창 번호), sub_index (하위 인덱스), prob (최적화 문제), obj_expression (목적 함수 표현식), cvx_error_msg  (CVX 오류 메시지)
        """
        TellUser.info(f'Optimization problem was {prob.status}')
        # save solver used
        try:
            self.solvers.append(prob.solver_stats.solver_name)
        except AttributeError:
            pass

        if (prob.status == 'infeasible') or (prob.status == 'unbounded') or (prob.status is None):
            # tell the user and throw an error specific to the problem being infeasible/unbounded
            error_msg = f'Optimization window {opt_window_num} was {prob.status}. No solution found. Look in *.log for for information'
            TellUser.error(cvx_error_msg)
            if prob.status == 'infeasible':
                raise SolverInfeasibleError(error_msg)
            elif prob.status == 'unbounded':
                raise SolverUnboundedError(error_msg)
            else:
                raise SolverError(error_msg)
        # evaluate optimal objective expression
        for cost, func in obj_expression.items():
            try:
                obj_expression[cost] = func.value
            except AttributeError:
                continue

        obj_values = pd.DataFrame(obj_expression, index=[opt_window_num])
        # then add objective expressions to financial obj_val
        self.objective_values = pd.concat([self.objective_values, obj_values])

        # GENERAL CHECK ON SOLUTION: check for non zero slack
        if np.any(abs(obj_values.filter(regex="_*slack$")) >= 1):
            TellUser.warning('non-zero slack variables found in optimization solution')
        for vs in self.service_agg.value_streams.values():
            vs.save_variable_results(sub_index)

        for der in self.poi.active_ders:
            # record the solution of the variables and run again
            der.save_variable_results(sub_index)
            # calculate degradation in Battery instances
            if der.tag == "Battery":
                der.calc_degradation(opt_window_num, sub_index[0], sub_index[-1])
