option = {
  aspectScale: 0.4,
    tooltip : {
        trigger: 'axis'
    },
    legend: {
        data:['LaSOT','TNL2K','UAV'],
        textStyle:{
          fontSize:20,
          fontWeight:'bolder'
        }
    },
    toolbox: {
        show : true,
        feature : {
            mark : {show: true},
            dataView : {show: true, readOnly: false},
            magicType : {show: true, type: ['line', 'bar', 'stack', 'tiled']},
            restore : {show: true},
            saveAsImage : {show: true}
        }
    },
    calculable : true,
    xAxis : [
        {
           name:'[α,β]',
            nameTextStyle:{
              fontSize:20,
              fontWeight:'bold'
            }, 
            type : 'category',
            boundaryGap : false,
            axisLabel: {
      interval: 0,
      fontSize: 20,// 设置X轴字体大小
    },
            data : ['[1,0]','[0.3,0.7]','[0.4,0.6]','[0.5,0.5]','[0.7,0.3]','[0,1]']
        }
    ],
    yAxis : [
        {
          name:'Success',
            nameTextStyle:{
              fontSize:20,
              fontWeight:'bold'
            },
            type : 'value',
            axisLabel: {
      interval: 0,
      fontSize: 20,// 设置y轴字体大小
    },
            min: 54 // 设置 y 轴的起始值为 54
        }
    ],
    
    series: [
    { type: "line",
        smooth: true,
        markLine: {
            symbol: 'none', // 去掉辅助线首尾圆点和箭头
            lineStyle: {
                color: '#CCCCCC', // 设置灰色
                type: 'dashed', // 设置为虚线
                width: 3 // 加粗线宽
            },
            data: [
                // 横向辅助线
                { yAxis: 62.77, label: { normal: { show: true, position: 'end', formatter: '{c}',fontSize:18 } } },
                { yAxis: 64.67, label: { normal: { show: true, position: 'end', formatter: '{c}', fontSize:18} } },
                { yAxis: 57.88, label: { normal: { show: true, position: 'end', formatter: '{c}', fontSize:18} } },
                // 竖向辅助线
                {
                    xAxis: 3,
                    label: {
                        normal: {
                            show: false // 不显示标签
                        }
                    }
                }
            ]
        },
        data: [], // 辅助线不需要数据
        symbolSize: 0, // 去掉辅助线的数据点
        itemStyle: {
            normal: {
                color: 'transparent' // 辅助线不需要显示数据点
            }
        }
    },
        {
            name:'LaSOT',
            type:'line',
            data:[61.01, 61.32, 62.08, 62.77, 60.88, 60.07],
            symbolSize:12,
            lineStyle:{
              width:4
            }
        },
        {
            name:'TNL2K',
            type:'line',
            data:[55.36, 56.46, 57.41, 57.88, 57.41, 55.35],
            symbolSize:12,
            lineStyle:{
              width:4
            }
        },
        {
            name:'UAV',
            type:'line',
            data:[64.49, 64.04, 64.11, 64.67, 64.21, 63.86],
            symbolSize:12,
            lineStyle:{
              width:4
            }
        }
    ]
};