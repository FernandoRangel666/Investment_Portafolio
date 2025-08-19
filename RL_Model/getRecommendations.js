// fetch-native.js

async function getRecommendations() {
  try {
    const response = await fetch('http://localhost:5000/recommendations');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    console.log('\n🏆 TOP 10 RECOMMENDATIONS\n');
    data.top_10_recommendations.forEach((stock, i) => {
      console.log(`${i+1}. ${stock.ticker} (${stock.name}) — ${stock.avg_allocation}%`);
    });

    console.log('\n💵 Profit Summary');
    console.log(`RL Agent: $${data.profit.rl_agent_final}`);
    console.log(`Buy & Hold: $${data.profit.buy_and_hold_final}`);
  } catch (err) {
    console.error('❌ Error fetching data:', err.message);
  }
}

getRecommendations();