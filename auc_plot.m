
load result.mat;
auc_result = [];
figure;

for i=1:919

  if sum(label(:,i)) == 0
   continue
  end

  [x, y, ~, auc] = perfcurve(label(:, i), result(:, i), 1);
  auc_result = [auc_result auc];

  plot(x, y);
  hold on;
end

print('auc_result.pdf', '-dpdf');
save('auc_result.mat', 'auc_result', '-v7.3');