import { Sidebar } from './sidebar.js';

export default {
  title: 'Sidebar',
  parameters: {
    chromatic: { disableSnapshot: false },
  },
};

const Template = (args) => new Sidebar(args);

export const Default = Template.bind({});
Default.args = {
  primary: true,
  label: 'Sidebar',
};
