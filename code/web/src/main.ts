import { loadGraphData, buildGraphology } from "./graphData";

async function bootstrap() {
  const data = await loadGraphData("/data/graph.json");
  const graph = buildGraphology(data);
  console.log(`loaded ${graph.order} nodes, ${graph.size} edges`);
}

bootstrap().catch((err) => {
  console.error(err);
});
